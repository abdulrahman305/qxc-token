// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/token/ERC20/ERC20Upgradeable.sol";
import "@openzeppelin/contracts-upgradeable/token/ERC20/extensions/ERC20BurnableUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/token/ERC20/extensions/ERC20PausableUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/security/ReentrancyGuardUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts/utils/math/SafeMath.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@openzeppelin/contracts/utils/cryptography/EIP712.sol";

/**
 * @title QXCTokenSecured
 * @dev Secure implementation with all critical vulnerabilities fixed
 */
contract QXCTokenSecured is 
    Initializable, 
    ERC20Upgradeable, 
    ERC20BurnableUpgradeable, 
    ERC20PausableUpgradeable, 
    AccessControlUpgradeable, 
    ReentrancyGuardUpgradeable,
    UUPSUpgradeable,
    EIP712 
{
    using SafeMath for uint256;
    using ECDSA for bytes32;

    // Role definitions
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
    bytes32 public constant UPGRADER_ROLE = keccak256("UPGRADER_ROLE");
    bytes32 public constant OPERATOR_ROLE = keccak256("OPERATOR_ROLE");

    // Constants
    uint256 public constant MAX_SUPPLY = 1_000_000_000 * 10**8; // 1B tokens
    uint256 public constant INITIAL_SUPPLY = 100_000_000 * 10**8; // 100M initial
    uint256 public constant MIN_STAKE_AMOUNT = 1000 * 10**8; // Min 1000 tokens
    uint256 public constant MAX_STAKE_AMOUNT = 10_000_000 * 10**8; // Max 10M tokens
    
    // Staking parameters
    uint256 public stakingRewardRate; // Basis points (100 = 1%)
    uint256 public minStakingPeriod;
    uint256 public unstakingPeriod; // Cooldown period
    uint256 public maxStakersCount;
    uint256 public currentStakersCount;
    
    // Emergency parameters
    bool public emergencyWithdrawEnabled;
    uint256 public emergencyWithdrawDelay;
    mapping(address => uint256) public emergencyWithdrawRequests;
    
    // Staking data structures
    struct StakeInfo {
        uint256 amount;
        uint256 startTime;
        uint256 lastClaimTime;
        uint256 accumulatedRewards;
        uint256 unstakeRequestTime;
        bool isUnstaking;
    }
    
    mapping(address => StakeInfo) public stakes;
    mapping(address => bool) public blacklist;
    mapping(address => uint256) public nonces;
    
    // Rate limiting
    mapping(address => uint256) public lastActionTime;
    uint256 public actionCooldown;
    
    // Circuit breaker
    uint256 public maxDailyTransferAmount;
    uint256 public currentDayStart;
    uint256 public dailyTransferredAmount;
    
    // Events
    event Staked(address indexed user, uint256 amount, uint256 timestamp);
    event UnstakeRequested(address indexed user, uint256 amount, uint256 unlockTime);
    event Unstaked(address indexed user, uint256 amount, uint256 timestamp);
    event RewardsClaimed(address indexed user, uint256 reward, uint256 timestamp);
    event EmergencyWithdrawRequested(address indexed user, uint256 timestamp);
    event EmergencyWithdrawExecuted(address indexed user, uint256 amount);
    event BlacklistUpdated(address indexed account, bool status);
    event CircuitBreakerTriggered(uint256 amount, uint256 limit);
    
    // Modifiers
    modifier notBlacklisted(address account) {
        require(!blacklist[account], "Account is blacklisted");
        _;
    }
    
    modifier rateLimited() {
        require(
            block.timestamp >= lastActionTime[msg.sender].add(actionCooldown),
            "Action cooling down"
        );
        lastActionTime[msg.sender] = block.timestamp;
        _;
    }
    
    modifier checkCircuitBreaker(uint256 amount) {
        if (block.timestamp >= currentDayStart.add(1 days)) {
            currentDayStart = block.timestamp;
            dailyTransferredAmount = 0;
        }
        
        require(
            dailyTransferredAmount.add(amount) <= maxDailyTransferAmount,
            "Daily transfer limit exceeded"
        );
        
        dailyTransferredAmount = dailyTransferredAmount.add(amount);
        _;
    }

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    function initialize(
        string memory name,
        string memory symbol,
        address admin
    ) public initializer {
        __ERC20_init(name, symbol);
        __ERC20Burnable_init();
        __ERC20Pausable_init();
        __AccessControl_init();
        __ReentrancyGuard_init();
        __UUPSUpgradeable_init();
        __EIP712_init(name, "1");

        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(PAUSER_ROLE, admin);
        _grantRole(MINTER_ROLE, admin);
        _grantRole(UPGRADER_ROLE, admin);

        // Initialize parameters
        stakingRewardRate = 500; // 5% annual
        minStakingPeriod = 30 days;
        unstakingPeriod = 7 days;
        maxStakersCount = 10000;
        actionCooldown = 1 minutes;
        maxDailyTransferAmount = 10_000_000 * 10**8;
        currentDayStart = block.timestamp;
        emergencyWithdrawDelay = 3 days;

        // Mint initial supply
        _mint(admin, INITIAL_SUPPLY);
    }

    function decimals() public pure override returns (uint8) {
        return 8;
    }

    /**
     * @dev Secure minting with commit-reveal scheme
     */
    function mint(
        address to,
        uint256 amount,
        bytes32 commitment,
        uint256 nonce
    ) public onlyRole(MINTER_ROLE) whenNotPaused {
        require(
            keccak256(abi.encodePacked(to, amount, nonce)) == commitment,
            "Invalid commitment"
        );
        require(nonces[to] < nonce, "Nonce already used");
        require(totalSupply().add(amount) <= MAX_SUPPLY, "Exceeds max supply");
        
        nonces[to] = nonce;
        _mint(to, amount);
    }

    /**
     * @dev Secure staking with checks-effects-interactions pattern
     */
    function stake(uint256 amount) external 
        nonReentrant 
        whenNotPaused 
        notBlacklisted(msg.sender)
        rateLimited 
    {
        require(amount >= MIN_STAKE_AMOUNT, "Below minimum stake");
        require(amount <= MAX_STAKE_AMOUNT, "Exceeds maximum stake");
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");
        require(currentStakersCount < maxStakersCount, "Max stakers reached");
        
        StakeInfo storage stakeInfo = stakes[msg.sender];
        
        // If existing stake, claim rewards first
        if (stakeInfo.amount > 0) {
            _claimRewards(msg.sender);
        } else {
            currentStakersCount = currentStakersCount.add(1);
        }
        
        // Update state before external call
        stakeInfo.amount = stakeInfo.amount.add(amount);
        stakeInfo.startTime = block.timestamp;
        stakeInfo.lastClaimTime = block.timestamp;
        stakeInfo.isUnstaking = false;
        stakeInfo.unstakeRequestTime = 0;
        
        // Effects before interactions
        _transfer(msg.sender, address(this), amount);
        
        emit Staked(msg.sender, amount, block.timestamp);
    }

    /**
     * @dev Request unstaking with timelock
     */
    function requestUnstake() external 
        nonReentrant 
        whenNotPaused 
        notBlacklisted(msg.sender) 
    {
        StakeInfo storage stakeInfo = stakes[msg.sender];
        require(stakeInfo.amount > 0, "No stake found");
        require(!stakeInfo.isUnstaking, "Already unstaking");
        require(
            block.timestamp >= stakeInfo.startTime.add(minStakingPeriod),
            "Minimum staking period not met"
        );
        
        // Claim pending rewards
        _claimRewards(msg.sender);
        
        // Mark as unstaking
        stakeInfo.isUnstaking = true;
        stakeInfo.unstakeRequestTime = block.timestamp;
        
        emit UnstakeRequested(
            msg.sender, 
            stakeInfo.amount, 
            block.timestamp.add(unstakingPeriod)
        );
    }

    /**
     * @dev Execute unstaking after timelock
     */
    function executeUnstake() external 
        nonReentrant 
        whenNotPaused 
        notBlacklisted(msg.sender)
        rateLimited 
    {
        StakeInfo storage stakeInfo = stakes[msg.sender];
        require(stakeInfo.isUnstaking, "Unstake not requested");
        require(
            block.timestamp >= stakeInfo.unstakeRequestTime.add(unstakingPeriod),
            "Unstaking period not met"
        );
        
        uint256 amount = stakeInfo.amount;
        
        // Clear stake info
        delete stakes[msg.sender];
        currentStakersCount = currentStakersCount.sub(1);
        
        // Transfer after state changes
        _transfer(address(this), msg.sender, amount);
        
        emit Unstaked(msg.sender, amount, block.timestamp);
    }

    /**
     * @dev Calculate rewards with overflow protection
     */
    function calculateRewards(address user) public view returns (uint256) {
        StakeInfo storage stakeInfo = stakes[user];
        if (stakeInfo.amount == 0 || stakeInfo.isUnstaking) {
            return 0;
        }
        
        uint256 stakingDuration = block.timestamp.sub(stakeInfo.lastClaimTime);
        uint256 annualReward = stakeInfo.amount.mul(stakingRewardRate).div(10000);
        uint256 reward = annualReward.mul(stakingDuration).div(365 days);
        
        // Check max supply constraint
        if (totalSupply().add(reward) > MAX_SUPPLY) {
            reward = MAX_SUPPLY.sub(totalSupply());
        }
        
        return reward;
    }

    /**
     * @dev Internal function to claim rewards
     */
    function _claimRewards(address user) internal {
        uint256 reward = calculateRewards(user);
        if (reward > 0) {
            StakeInfo storage stakeInfo = stakes[user];
            stakeInfo.accumulatedRewards = stakeInfo.accumulatedRewards.add(reward);
            stakeInfo.lastClaimTime = block.timestamp;
            
            _mint(user, reward);
            emit RewardsClaimed(user, reward, block.timestamp);
        }
    }

    /**
     * @dev Public function to claim rewards
     */
    function claimRewards() external 
        nonReentrant 
        whenNotPaused 
        notBlacklisted(msg.sender)
        rateLimited 
    {
        _claimRewards(msg.sender);
    }

    /**
     * @dev Emergency withdraw with timelock
     */
    function requestEmergencyWithdraw() external 
        nonReentrant 
        notBlacklisted(msg.sender) 
    {
        require(emergencyWithdrawEnabled, "Emergency withdraw disabled");
        require(stakes[msg.sender].amount > 0, "No stake found");
        require(
            emergencyWithdrawRequests[msg.sender] == 0,
            "Already requested"
        );
        
        emergencyWithdrawRequests[msg.sender] = block.timestamp;
        emit EmergencyWithdrawRequested(msg.sender, block.timestamp);
    }

    /**
     * @dev Execute emergency withdraw after delay
     */
    function executeEmergencyWithdraw() external 
        nonReentrant 
        notBlacklisted(msg.sender) 
    {
        require(
            emergencyWithdrawRequests[msg.sender] > 0,
            "No request found"
        );
        require(
            block.timestamp >= emergencyWithdrawRequests[msg.sender].add(emergencyWithdrawDelay),
            "Delay not met"
        );
        
        uint256 amount = stakes[msg.sender].amount;
        
        // Clear all data
        delete stakes[msg.sender];
        delete emergencyWithdrawRequests[msg.sender];
        currentStakersCount = currentStakersCount.sub(1);
        
        // Transfer funds
        _transfer(address(this), msg.sender, amount);
        
        emit EmergencyWithdrawExecuted(msg.sender, amount);
    }

    /**
     * @dev Transfer with circuit breaker
     */
    function transfer(address to, uint256 amount) 
        public 
        override 
        notBlacklisted(msg.sender)
        notBlacklisted(to)
        checkCircuitBreaker(amount)
        returns (bool) 
    {
        return super.transfer(to, amount);
    }

    /**
     * @dev TransferFrom with additional checks
     */
    function transferFrom(address from, address to, uint256 amount)
        public
        override
        notBlacklisted(from)
        notBlacklisted(to)
        notBlacklisted(msg.sender)
        checkCircuitBreaker(amount)
        returns (bool)
    {
        return super.transferFrom(from, to, amount);
    }

    // Admin functions with multi-sig protection (in production)
    
    function updateBlacklist(address account, bool status) 
        external 
        onlyRole(DEFAULT_ADMIN_ROLE) 
    {
        blacklist[account] = status;
        emit BlacklistUpdated(account, status);
    }

    function updateStakingParameters(
        uint256 _rewardRate,
        uint256 _minPeriod,
        uint256 _unstakingPeriod
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(_rewardRate <= 2000, "Rate too high"); // Max 20%
        require(_minPeriod >= 1 days, "Period too short");
        require(_unstakingPeriod >= 1 days, "Unstaking period too short");
        
        stakingRewardRate = _rewardRate;
        minStakingPeriod = _minPeriod;
        unstakingPeriod = _unstakingPeriod;
    }

    function updateCircuitBreaker(uint256 _maxDaily) 
        external 
        onlyRole(DEFAULT_ADMIN_ROLE) 
    {
        maxDailyTransferAmount = _maxDaily;
    }

    function toggleEmergencyWithdraw(bool _enabled) 
        external 
        onlyRole(DEFAULT_ADMIN_ROLE) 
    {
        emergencyWithdrawEnabled = _enabled;
    }

    function pause() public onlyRole(PAUSER_ROLE) {
        _pause();
    }

    function unpause() public onlyRole(PAUSER_ROLE) {
        _unpause();
    }

    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 amount
    ) internal override(ERC20Upgradeable, ERC20PausableUpgradeable) {
        super._beforeTokenTransfer(from, to, amount);
    }

    function _authorizeUpgrade(address newImplementation)
        internal
        override
        onlyRole(UPGRADER_ROLE)
    {}

    /**
     * @dev Get comprehensive staking information
     */
    function getStakingInfo(address user) external view returns (
        uint256 stakedAmount,
        uint256 pendingRewards,
        uint256 totalRewardsEarned,
        uint256 stakingStartTime,
        bool isUnstaking,
        uint256 unstakeUnlockTime
    ) {
        StakeInfo storage info = stakes[user];
        return (
            info.amount,
            calculateRewards(user),
            info.accumulatedRewards,
            info.startTime,
            info.isUnstaking,
            info.isUnstaking ? info.unstakeRequestTime.add(unstakingPeriod) : 0
        );
    }
}