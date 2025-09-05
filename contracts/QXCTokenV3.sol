// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/token/ERC20/ERC20Upgradeable.sol";
import "@openzeppelin/contracts-upgradeable/token/ERC20/extensions/ERC20BurnableUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/token/ERC20/extensions/ERC20SnapshotUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/security/PausableUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/security/ReentrancyGuardUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";

/**
 * @title QXC Token V3 - Enterprise Grade Financial Token
 * @notice Production-ready token with advanced DeFi features and regulatory compliance
 * @dev Implements comprehensive security, governance, and financial features
 */
contract QXCTokenV3 is 
    Initializable,
    ERC20Upgradeable,
    ERC20BurnableUpgradeable,
    ERC20SnapshotUpgradeable,
    AccessControlUpgradeable,
    PausableUpgradeable,
    ReentrancyGuardUpgradeable,
    UUPSUpgradeable
{
    // Role definitions
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
    bytes32 public constant SNAPSHOT_ROLE = keccak256("SNAPSHOT_ROLE");
    bytes32 public constant COMPLIANCE_ROLE = keccak256("COMPLIANCE_ROLE");
    bytes32 public constant UPGRADER_ROLE = keccak256("UPGRADER_ROLE");
    bytes32 public constant TREASURY_ROLE = keccak256("TREASURY_ROLE");

    // Token economics
    uint256 public constant MAX_SUPPLY = 1_000_000_000 * 10**18; // 1 billion tokens
    uint256 public constant INITIAL_SUPPLY = 100_000_000 * 10**18; // 100 million initial
    
    // Fee structure (basis points: 100 = 1%)
    uint256 public transferFee;
    uint256 public stakingFee;
    uint256 public constant MAX_FEE = 500; // 5% maximum fee
    uint256 public constant FEE_DENOMINATOR = 10000;
    
    // Staking mechanism
    struct StakeInfo {
        uint256 amount;
        uint256 timestamp;
        uint256 reward;
        uint256 lockPeriod;
        bool isActive;
    }
    
    mapping(address => StakeInfo[]) public stakes;
    mapping(address => uint256) public totalStaked;
    uint256 public totalStakedSupply;
    uint256 public stakingAPR; // Annual Percentage Rate in basis points
    uint256 public minStakingAmount;
    uint256 public maxStakingAmount;
    
    // Governance
    struct Proposal {
        uint256 id;
        address proposer;
        string description;
        uint256 forVotes;
        uint256 againstVotes;
        uint256 startTime;
        uint256 endTime;
        bool executed;
        mapping(address => bool) hasVoted;
    }
    
    mapping(uint256 => Proposal) public proposals;
    uint256 public proposalCount;
    uint256 public votingPeriod;
    uint256 public proposalThreshold;
    
    // Compliance and regulatory
    mapping(address => bool) public blacklisted;
    mapping(address => bool) public whitelisted;
    mapping(address => uint256) public kycLevel; // 0: None, 1: Basic, 2: Enhanced, 3: Institutional
    mapping(address => string) public countryCode;
    mapping(string => bool) public restrictedCountries;
    bool public whitelistEnabled;
    bool public kycRequired;
    
    // Transaction limits
    mapping(address => uint256) public dailyTransferLimit;
    mapping(address => uint256) public dailyTransferred;
    mapping(address => uint256) public lastTransferDate;
    uint256 public defaultDailyLimit;
    uint256 public maxTransactionAmount;
    
    // Treasury and reserves
    address public treasuryWallet;
    address public reserveWallet;
    uint256 public treasuryBalance;
    uint256 public reserveBalance;
    
    // Circuit breaker
    bool public emergencyStop;
    uint256 public emergencyStopTime;
    uint256 public constant EMERGENCY_DURATION = 24 hours;
    
    // Events
    event Staked(address indexed user, uint256 amount, uint256 lockPeriod);
    event Unstaked(address indexed user, uint256 amount, uint256 reward);
    event RewardClaimed(address indexed user, uint256 reward);
    event ProposalCreated(uint256 indexed proposalId, address proposer, string description);
    event Voted(uint256 indexed proposalId, address voter, bool support, uint256 votes);
    event ProposalExecuted(uint256 indexed proposalId);
    event BlacklistUpdated(address indexed account, bool status);
    event WhitelistUpdated(address indexed account, bool status);
    event KYCLevelUpdated(address indexed account, uint256 level);
    event FeeCollected(address indexed from, uint256 amount, string feeType);
    event EmergencyStopActivated(address indexed activator, uint256 timestamp);
    event EmergencyStopDeactivated(address indexed deactivator, uint256 timestamp);
    event TreasuryWithdrawal(address indexed to, uint256 amount);
    
    // Modifiers
    modifier notBlacklisted(address account) {
        require(!blacklisted[account], "Account is blacklisted");
        _;
    }
    
    modifier checkWhitelist(address from, address to) {
        if (whitelistEnabled) {
            require(
                whitelisted[from] || whitelisted[to] || 
                hasRole(DEFAULT_ADMIN_ROLE, from) || hasRole(DEFAULT_ADMIN_ROLE, to),
                "Whitelist required"
            );
        }
        _;
    }
    
    modifier checkKYC(address account) {
        if (kycRequired) {
            require(kycLevel[account] > 0, "KYC verification required");
        }
        _;
    }
    
    modifier checkTransferLimits(address sender, uint256 amount) {
        require(amount <= maxTransactionAmount, "Exceeds max transaction amount");
        
        uint256 today = block.timestamp / 1 days;
        if (lastTransferDate[sender] < today) {
            dailyTransferred[sender] = 0;
            lastTransferDate[sender] = today;
        }
        
        uint256 limit = dailyTransferLimit[sender] > 0 ? dailyTransferLimit[sender] : defaultDailyLimit;
        require(dailyTransferred[sender] + amount <= limit, "Exceeds daily limit");
        dailyTransferred[sender] += amount;
        _;
    }
    
    modifier notEmergencyStopped() {
        require(!emergencyStop || block.timestamp > emergencyStopTime + EMERGENCY_DURATION, 
                "Emergency stop active");
        _;
    }

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    /**
     * @dev Initialize the contract
     */
    function initialize(
        address _treasuryWallet,
        address _reserveWallet
    ) public initializer {
        __ERC20_init("QENEX Token", "QXC");
        __ERC20Burnable_init();
        __ERC20Snapshot_init();
        __AccessControl_init();
        __Pausable_init();
        __ReentrancyGuard_init();
        __UUPSUpgradeable_init();

        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(MINTER_ROLE, msg.sender);
        _grantRole(PAUSER_ROLE, msg.sender);
        _grantRole(SNAPSHOT_ROLE, msg.sender);
        _grantRole(COMPLIANCE_ROLE, msg.sender);
        _grantRole(UPGRADER_ROLE, msg.sender);
        _grantRole(TREASURY_ROLE, msg.sender);

        treasuryWallet = _treasuryWallet;
        reserveWallet = _reserveWallet;
        
        // Initial configuration
        transferFee = 10; // 0.1%
        stakingFee = 5; // 0.05%
        stakingAPR = 500; // 5% APR
        minStakingAmount = 100 * 10**18; // 100 tokens minimum
        maxStakingAmount = 1_000_000 * 10**18; // 1M tokens maximum
        votingPeriod = 3 days;
        proposalThreshold = 10_000 * 10**18; // 10,000 tokens to create proposal
        defaultDailyLimit = 100_000 * 10**18; // 100,000 tokens
        maxTransactionAmount = 10_000 * 10**18; // 10,000 tokens
        
        // Mint initial supply
        _mint(treasuryWallet, INITIAL_SUPPLY * 60 / 100); // 60% to treasury
        _mint(reserveWallet, INITIAL_SUPPLY * 40 / 100); // 40% to reserve
        
        treasuryBalance = INITIAL_SUPPLY * 60 / 100;
        reserveBalance = INITIAL_SUPPLY * 40 / 100;
    }

    /**
     * @dev Transfer with fee mechanism
     */
    function transfer(address to, uint256 amount) 
        public 
        override 
        notBlacklisted(msg.sender)
        notBlacklisted(to)
        checkWhitelist(msg.sender, to)
        checkKYC(msg.sender)
        checkTransferLimits(msg.sender, amount)
        notEmergencyStopped
        whenNotPaused
        returns (bool) 
    {
        uint256 fee = (amount * transferFee) / FEE_DENOMINATOR;
        uint256 amountAfterFee = amount - fee;
        
        if (fee > 0) {
            super.transfer(treasuryWallet, fee);
            treasuryBalance += fee;
            emit FeeCollected(msg.sender, fee, "TRANSFER");
        }
        
        return super.transfer(to, amountAfterFee);
    }

    /**
     * @dev Stake tokens for rewards
     */
    function stake(uint256 amount, uint256 lockPeriod) 
        external 
        nonReentrant 
        notEmergencyStopped
        whenNotPaused 
    {
        require(amount >= minStakingAmount, "Below minimum stake");
        require(amount <= maxStakingAmount, "Above maximum stake");
        require(lockPeriod >= 30 days, "Minimum lock period 30 days");
        require(lockPeriod <= 365 days, "Maximum lock period 365 days");
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");
        
        // Transfer tokens to contract
        _transfer(msg.sender, address(this), amount);
        
        // Apply staking fee
        uint256 fee = (amount * stakingFee) / FEE_DENOMINATOR;
        uint256 stakedAmount = amount - fee;
        
        if (fee > 0) {
            _transfer(address(this), treasuryWallet, fee);
            treasuryBalance += fee;
            emit FeeCollected(msg.sender, fee, "STAKING");
        }
        
        // Record stake
        stakes[msg.sender].push(StakeInfo({
            amount: stakedAmount,
            timestamp: block.timestamp,
            reward: 0,
            lockPeriod: lockPeriod,
            isActive: true
        }));
        
        totalStaked[msg.sender] += stakedAmount;
        totalStakedSupply += stakedAmount;
        
        emit Staked(msg.sender, stakedAmount, lockPeriod);
    }

    /**
     * @dev Unstake tokens and claim rewards
     */
    function unstake(uint256 stakeIndex) 
        external 
        nonReentrant 
        notEmergencyStopped
        whenNotPaused 
    {
        require(stakeIndex < stakes[msg.sender].length, "Invalid stake index");
        StakeInfo storage stakeInfo = stakes[msg.sender][stakeIndex];
        require(stakeInfo.isActive, "Stake already withdrawn");
        require(
            block.timestamp >= stakeInfo.timestamp + stakeInfo.lockPeriod,
            "Lock period not expired"
        );
        
        // Calculate rewards
        uint256 reward = calculateReward(msg.sender, stakeIndex);
        uint256 totalAmount = stakeInfo.amount + reward;
        
        // Update state
        stakeInfo.isActive = false;
        totalStaked[msg.sender] -= stakeInfo.amount;
        totalStakedSupply -= stakeInfo.amount;
        
        // Transfer tokens
        _transfer(address(this), msg.sender, totalAmount);
        
        emit Unstaked(msg.sender, stakeInfo.amount, reward);
    }

    /**
     * @dev Calculate staking reward
     */
    function calculateReward(address user, uint256 stakeIndex) 
        public 
        view 
        returns (uint256) 
    {
        StakeInfo memory stakeInfo = stakes[user][stakeIndex];
        if (!stakeInfo.isActive) return 0;
        
        uint256 stakingDuration = block.timestamp - stakeInfo.timestamp;
        uint256 annualReward = (stakeInfo.amount * stakingAPR) / FEE_DENOMINATOR;
        uint256 reward = (annualReward * stakingDuration) / 365 days;
        
        // Bonus for longer lock periods
        if (stakeInfo.lockPeriod >= 180 days) {
            reward = (reward * 120) / 100; // 20% bonus
        } else if (stakeInfo.lockPeriod >= 90 days) {
            reward = (reward * 110) / 100; // 10% bonus
        }
        
        return reward;
    }

    /**
     * @dev Create governance proposal
     */
    function createProposal(string memory description) 
        external 
        whenNotPaused 
    {
        require(balanceOf(msg.sender) >= proposalThreshold, "Insufficient tokens for proposal");
        
        proposalCount++;
        Proposal storage proposal = proposals[proposalCount];
        proposal.id = proposalCount;
        proposal.proposer = msg.sender;
        proposal.description = description;
        proposal.startTime = block.timestamp;
        proposal.endTime = block.timestamp + votingPeriod;
        
        emit ProposalCreated(proposalCount, msg.sender, description);
    }

    /**
     * @dev Vote on proposal
     */
    function vote(uint256 proposalId, bool support) 
        external 
        whenNotPaused 
    {
        Proposal storage proposal = proposals[proposalId];
        require(block.timestamp >= proposal.startTime, "Voting not started");
        require(block.timestamp <= proposal.endTime, "Voting ended");
        require(!proposal.hasVoted[msg.sender], "Already voted");
        
        uint256 votes = balanceOf(msg.sender);
        require(votes > 0, "No voting power");
        
        proposal.hasVoted[msg.sender] = true;
        
        if (support) {
            proposal.forVotes += votes;
        } else {
            proposal.againstVotes += votes;
        }
        
        emit Voted(proposalId, msg.sender, support, votes);
    }

    /**
     * @dev Mint new tokens (controlled supply)
     */
    function mint(address to, uint256 amount) 
        external 
        onlyRole(MINTER_ROLE) 
    {
        require(totalSupply() + amount <= MAX_SUPPLY, "Exceeds max supply");
        _mint(to, amount);
    }

    /**
     * @dev Update compliance status
     */
    function updateKYCLevel(address account, uint256 level) 
        external 
        onlyRole(COMPLIANCE_ROLE) 
    {
        require(level <= 3, "Invalid KYC level");
        kycLevel[account] = level;
        
        // Auto-whitelist high KYC levels
        if (level >= 2) {
            whitelisted[account] = true;
        }
        
        emit KYCLevelUpdated(account, level);
    }

    /**
     * @dev Emergency stop mechanism
     */
    function activateEmergencyStop() 
        external 
        onlyRole(PAUSER_ROLE) 
    {
        emergencyStop = true;
        emergencyStopTime = block.timestamp;
        emit EmergencyStopActivated(msg.sender, block.timestamp);
    }

    /**
     * @dev Deactivate emergency stop
     */
    function deactivateEmergencyStop() 
        external 
        onlyRole(DEFAULT_ADMIN_ROLE) 
    {
        emergencyStop = false;
        emit EmergencyStopDeactivated(msg.sender, block.timestamp);
    }

    /**
     * @dev Withdraw from treasury
     */
    function withdrawFromTreasury(address to, uint256 amount) 
        external 
        onlyRole(TREASURY_ROLE) 
        nonReentrant 
    {
        require(amount <= treasuryBalance, "Insufficient treasury balance");
        treasuryBalance -= amount;
        _transfer(treasuryWallet, to, amount);
        emit TreasuryWithdrawal(to, amount);
    }

    /**
     * @dev Create snapshot for governance
     */
    function snapshot() 
        external 
        onlyRole(SNAPSHOT_ROLE) 
        returns (uint256) 
    {
        return _snapshot();
    }

    /**
     * @dev Pause token transfers
     */
    function pause() external onlyRole(PAUSER_ROLE) {
        _pause();
    }

    /**
     * @dev Unpause token transfers  
     */
    function unpause() external onlyRole(PAUSER_ROLE) {
        _unpause();
    }

    /**
     * @dev Authorize upgrade
     */
    function _authorizeUpgrade(address newImplementation)
        internal
        onlyRole(UPGRADER_ROLE)
        override
    {}

    /**
     * @dev Override required by Solidity
     */
    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 amount
    ) internal override(ERC20Upgradeable, ERC20SnapshotUpgradeable) whenNotPaused {
        super._beforeTokenTransfer(from, to, amount);
    }
}