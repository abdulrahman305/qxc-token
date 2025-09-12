// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title QXC Token - QENEX Native Token
 * @dev Advanced ERC-20 implementation with governance, staking, and DeFi features
 */

interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

interface IERC20Metadata is IERC20 {
    function name() external view returns (string memory);
    function symbol() external view returns (string memory);
    function decimals() external view returns (uint8);
}

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
    
    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }
}

abstract contract Ownable is Context {
    address private _owner;
    
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    
    constructor() {
        _transferOwnership(_msgSender());
    }
    
    modifier onlyOwner() {
        _checkOwner();
        _;
    }
    
    function owner() public view virtual returns (address) {
        return _owner;
    }
    
    function _checkOwner() internal view virtual {
        require(owner() == _msgSender(), "Ownable: caller is not the owner");
    }
    
    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }
    
    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        _transferOwnership(newOwner);
    }
    
    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

abstract contract Pausable is Context {
    event Paused(address account);
    event Unpaused(address account);
    
    bool private _paused;
    
    constructor() {
        _paused = false;
    }
    
    modifier whenNotPaused() {
        _requireNotPaused();
        _;
    }
    
    modifier whenPaused() {
        _requirePaused();
        _;
    }
    
    function paused() public view virtual returns (bool) {
        return _paused;
    }
    
    function _requireNotPaused() internal view virtual {
        require(!paused(), "Pausable: paused");
    }
    
    function _requirePaused() internal view virtual {
        require(paused(), "Pausable: not paused");
    }
    
    function _pause() internal virtual whenNotPaused {
        _paused = true;
        emit Paused(_msgSender());
    }
    
    function _unpause() internal virtual whenPaused {
        _paused = false;
        emit Unpaused(_msgSender());
    }
}

contract QXCToken is Context, IERC20, IERC20Metadata, Ownable, Pausable {
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    
    // Staking variables
    mapping(address => uint256) public stakedBalance;
    mapping(address => uint256) public stakingTimestamp;
    mapping(address => uint256) public accumulatedRewards;
    
    // Governance variables
    mapping(address => uint256) public votingPower;
    mapping(uint256 => Proposal) public proposals;
    mapping(uint256 => mapping(address => bool)) public hasVoted;
    uint256 public proposalCount;
    
    // DeFi integration variables
    mapping(address => bool) public authorizedPools;
    mapping(address => bool) public blacklisted;
    
    uint256 private _totalSupply;
    string public name;
    string public symbol;
    uint8 public decimals;
    
    // Economic parameters
    uint256 public constant MAX_SUPPLY = 1000000000 * 10**18; // 1 billion tokens
    uint256 public stakingAPY = 1200; // 12% APY (basis points)
    uint256 public transactionFee = 30; // 0.3% fee (basis points)
    uint256 public burnRate = 10; // 0.1% burn rate (basis points)
    
    // Treasury and reserves
    address public treasury;
    address public stakingRewards;
    address public liquidityReserve;
    
    uint256 public totalStaked;
    uint256 public totalBurned;
    
    struct Proposal {
        uint256 id;
        address proposer;
        string description;
        uint256 forVotes;
        uint256 againstVotes;
        uint256 startTime;
        uint256 endTime;
        uint256 executionTime; // SECURITY FIX: Add timelock
        bool executed;
        ProposalType proposalType;
        bytes callData;
    }
    
    enum ProposalType {
        ParameterChange,
        TreasuryAllocation,
        PoolAuthorization,
        EmergencyAction
    }
    
    event Staked(address indexed user, uint256 amount);
    event Unstaked(address indexed user, uint256 amount, uint256 rewards);
    event ProposalCreated(uint256 indexed proposalId, address indexed proposer, string description);
    event Voted(uint256 indexed proposalId, address indexed voter, bool support, uint256 weight);
    event ProposalExecuted(uint256 indexed proposalId);
    event PoolAuthorized(address indexed pool, bool authorized);
    event BlacklistUpdated(address indexed account, bool blacklisted);
    
    constructor() {
        name = "QENEX Coin";
        symbol = "QXC";
        decimals = 18;
        
        _totalSupply = MAX_SUPPLY;
        _balances[_msgSender()] = _totalSupply * 40 / 100; // 40% to deployer
        
        treasury = address(0x1234567890123456789012345678901234567890);
        stakingRewards = address(0x2345678901234567890123456789012345678901);
        liquidityReserve = address(0x3456789012345678901234567890123456789012);
        
        _balances[treasury] = _totalSupply * 30 / 100; // 30% to treasury
        _balances[stakingRewards] = _totalSupply * 20 / 100; // 20% for staking rewards
        _balances[liquidityReserve] = _totalSupply * 10 / 100; // 10% for liquidity
        
        emit Transfer(address(0), _msgSender(), _balances[_msgSender()]);
        emit Transfer(address(0), treasury, _balances[treasury]);
        emit Transfer(address(0), stakingRewards, _balances[stakingRewards]);
        emit Transfer(address(0), liquidityReserve, _balances[liquidityReserve]);
    }
    
    function totalSupply() public view virtual override returns (uint256) {
        return _totalSupply - totalBurned;
    }
    
    function balanceOf(address account) public view virtual override returns (uint256) {
        return _balances[account];
    }
    
    function transfer(address to, uint256 amount) public virtual override whenNotPaused returns (bool) {
        address owner = _msgSender();
        _transfer(owner, to, amount);
        return true;
    }
    
    function allowance(address owner, address spender) public view virtual override returns (uint256) {
        return _allowances[owner][spender];
    }
    
    function approve(address spender, uint256 amount) public virtual override whenNotPaused returns (bool) {
        address owner = _msgSender();
        _approve(owner, spender, amount);
        return true;
    }
    
    function transferFrom(
        address from,
        address to,
        uint256 amount
    ) public virtual override whenNotPaused returns (bool) {
        address spender = _msgSender();
        _spendAllowance(from, spender, amount);
        _transfer(from, to, amount);
        return true;
    }
    
    function _transfer(
        address from,
        address to,
        uint256 amount
    ) internal virtual {
        require(from != address(0), "ERC20: transfer from the zero address");
        require(to != address(0), "ERC20: transfer to the zero address");
        require(!blacklisted[from] && !blacklisted[to], "Address is blacklisted");
        
        uint256 fromBalance = _balances[from];
        require(fromBalance >= amount, "ERC20: transfer amount exceeds balance");
        
        // Calculate fees
        uint256 fee = 0;
        uint256 burnAmount = 0;
        
        if (!authorizedPools[from] && !authorizedPools[to]) {
            fee = amount * transactionFee / 10000;
            burnAmount = amount * burnRate / 10000;
        }
        
        uint256 transferAmount = amount - fee - burnAmount;
        
        unchecked {
            _balances[from] = fromBalance - amount;
            _balances[to] += transferAmount;
        }
        
        if (fee > 0) {
            _balances[treasury] += fee;
            emit Transfer(from, treasury, fee);
        }
        
        if (burnAmount > 0) {
            totalBurned += burnAmount;
            emit Transfer(from, address(0), burnAmount);
        }
        
        emit Transfer(from, to, transferAmount);
    }
    
    function _approve(
        address owner,
        address spender,
        uint256 amount
    ) internal virtual {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        
        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }
    
    function _spendAllowance(
        address owner,
        address spender,
        uint256 amount
    ) internal virtual {
        uint256 currentAllowance = allowance(owner, spender);
        if (currentAllowance != type(uint256).max) {
            require(currentAllowance >= amount, "ERC20: insufficient allowance");
            unchecked {
                _approve(owner, spender, currentAllowance - amount);
            }
        }
    }
    
    // Staking functions
    function stake(uint256 amount) external whenNotPaused {
        require(amount > 0, "Cannot stake 0 tokens");
        require(_balances[_msgSender()] >= amount, "Insufficient balance");
        
        // Claim pending rewards first
        if (stakedBalance[_msgSender()] > 0) {
            _claimRewards(_msgSender());
        }
        
        _transfer(_msgSender(), address(this), amount);
        
        stakedBalance[_msgSender()] += amount;
        stakingTimestamp[_msgSender()] = block.timestamp;
        totalStaked += amount;
        
        // Update voting power
        votingPower[_msgSender()] = stakedBalance[_msgSender()];
        
        emit Staked(_msgSender(), amount);
    }
    
    function unstake(uint256 amount) external whenNotPaused {
        require(amount > 0, "Cannot unstake 0 tokens");
        require(stakedBalance[_msgSender()] >= amount, "Insufficient staked balance");
        
        // Calculate and claim rewards
        uint256 rewards = calculateRewards(_msgSender());
        
        stakedBalance[_msgSender()] -= amount;
        totalStaked -= amount;
        
        // Update voting power
        votingPower[_msgSender()] = stakedBalance[_msgSender()];
        
        // Transfer staked tokens back
        _balances[address(this)] -= amount;
        _balances[_msgSender()] += amount;
        emit Transfer(address(this), _msgSender(), amount);
        
        // Transfer rewards
        if (rewards > 0) {
            _transferRewards(_msgSender(), rewards);
        }
        
        stakingTimestamp[_msgSender()] = block.timestamp;
        
        emit Unstaked(_msgSender(), amount, rewards);
    }
    
    function calculateRewards(address account) public view returns (uint256) {
        if (stakedBalance[account] == 0) {
            return 0;
        }
        
        uint256 stakingDuration = block.timestamp - stakingTimestamp[account];
        uint256 rewards = (stakedBalance[account] * stakingAPY * stakingDuration) / (365 days * 10000);
        
        return rewards + accumulatedRewards[account];
    }
    
    function claimRewards() external whenNotPaused {
        _claimRewards(_msgSender());
    }
    
    function _claimRewards(address account) internal {
        uint256 rewards = calculateRewards(account);
        
        if (rewards > 0) {
            accumulatedRewards[account] = 0;
            stakingTimestamp[account] = block.timestamp;
            _transferRewards(account, rewards);
        }
    }
    
    function _transferRewards(address to, uint256 amount) internal {
        require(_balances[stakingRewards] >= amount, "Insufficient reward balance");
        
        _balances[stakingRewards] -= amount;
        _balances[to] += amount;
        
        emit Transfer(stakingRewards, to, amount);
    }
    
    // Governance functions
    function createProposal(
        string memory description,
        ProposalType proposalType,
        bytes memory callData
    ) external returns (uint256) {
        require(votingPower[_msgSender()] >= 1000 * 10**18, "Insufficient voting power");
        
        uint256 proposalId = proposalCount++;
        
        proposals[proposalId] = Proposal({
            id: proposalId,
            proposer: _msgSender(),
            description: description,
            forVotes: 0,
            againstVotes: 0,
            startTime: block.timestamp,
            endTime: block.timestamp + 3 days,
            executionTime: block.timestamp + 3 days + 2 days, // SECURITY FIX: 2 day timelock after voting ends
            executed: false,
            proposalType: proposalType,
            callData: callData
        });
        
        emit ProposalCreated(proposalId, _msgSender(), description);
        
        return proposalId;
    }
    
    function vote(uint256 proposalId, bool support) external {
        Proposal storage proposal = proposals[proposalId];
        
        require(block.timestamp >= proposal.startTime, "Voting not started");
        require(block.timestamp <= proposal.endTime, "Voting ended");
        require(!hasVoted[proposalId][_msgSender()], "Already voted");
        require(votingPower[_msgSender()] > 0, "No voting power");
        
        hasVoted[proposalId][_msgSender()] = true;
        
        if (support) {
            proposal.forVotes += votingPower[_msgSender()];
        } else {
            proposal.againstVotes += votingPower[_msgSender()];
        }
        
        emit Voted(proposalId, _msgSender(), support, votingPower[_msgSender()]);
    }
    
    function executeProposal(uint256 proposalId) external {
        Proposal storage proposal = proposals[proposalId];
        
        require(block.timestamp > proposal.endTime, "Voting not ended");
        require(block.timestamp >= proposal.executionTime, "Timelock not expired"); // SECURITY FIX: Timelock check
        require(!proposal.executed, "Already executed");
        require(proposal.forVotes > proposal.againstVotes, "Proposal failed");
        require(proposal.forVotes >= (totalSupply() * 10) / 100, "Insufficient quorum"); // SECURITY FIX: Minimum quorum
        
        proposal.executed = true;
        
        // Execute based on proposal type
        if (proposal.proposalType == ProposalType.ParameterChange) {
            _executeParameterChange(proposal.callData);
        } else if (proposal.proposalType == ProposalType.TreasuryAllocation) {
            _executeTreasuryAllocation(proposal.callData);
        } else if (proposal.proposalType == ProposalType.PoolAuthorization) {
            _executePoolAuthorization(proposal.callData);
        } else if (proposal.proposalType == ProposalType.EmergencyAction) {
            _executeEmergencyAction(proposal.callData);
        }
        
        emit ProposalExecuted(proposalId);
    }
    
    function _executeParameterChange(bytes memory data) internal {
        (uint256 newAPY, uint256 newFee, uint256 newBurnRate) = abi.decode(data, (uint256, uint256, uint256));
        
        if (newAPY > 0 && newAPY <= 5000) stakingAPY = newAPY;
        if (newFee <= 100) transactionFee = newFee;
        if (newBurnRate <= 50) burnRate = newBurnRate;
    }
    
    function _executeTreasuryAllocation(bytes memory data) internal {
        (address recipient, uint256 amount) = abi.decode(data, (address, uint256));
        
        require(_balances[treasury] >= amount, "Insufficient treasury balance");
        
        _balances[treasury] -= amount;
        _balances[recipient] += amount;
        
        emit Transfer(treasury, recipient, amount);
    }
    
    function _executePoolAuthorization(bytes memory data) internal {
        (address pool, bool authorized) = abi.decode(data, (address, bool));
        
        authorizedPools[pool] = authorized;
        
        emit PoolAuthorized(pool, authorized);
    }
    
    function _executeEmergencyAction(bytes memory data) internal {
        (bool pause) = abi.decode(data, (bool));
        
        if (pause) {
            _pause();
        } else {
            _unpause();
        }
    }
    
    // Admin functions
    function authorizePool(address pool, bool authorized) external onlyOwner {
        authorizedPools[pool] = authorized;
        emit PoolAuthorized(pool, authorized);
    }
    
    function updateBlacklist(address account, bool blacklist) external onlyOwner {
        blacklisted[account] = blacklist;
        emit BlacklistUpdated(account, blacklist);
    }
    
    function setTreasury(address _treasury) external onlyOwner {
        require(_treasury != address(0), "Invalid treasury address");
        treasury = _treasury;
    }
    
    function setStakingRewards(address _stakingRewards) external onlyOwner {
        require(_stakingRewards != address(0), "Invalid staking rewards address");
        stakingRewards = _stakingRewards;
    }
    
    function setLiquidityReserve(address _liquidityReserve) external onlyOwner {
        require(_liquidityReserve != address(0), "Invalid liquidity reserve address");
        liquidityReserve = _liquidityReserve;
    }
    
    function pause() external onlyOwner {
        _pause();
    }
    
    function unpause() external onlyOwner {
        _unpause();
    }
    
    // View functions
    function getProposal(uint256 proposalId) external view returns (
        address proposer,
        string memory description,
        uint256 forVotes,
        uint256 againstVotes,
        uint256 startTime,
        uint256 endTime,
        uint256 executionTime, // SECURITY FIX: Include execution time in view
        bool executed
    ) {
        Proposal memory proposal = proposals[proposalId];
        return (
            proposal.proposer,
            proposal.description,
            proposal.forVotes,
            proposal.againstVotes,
            proposal.startTime,
            proposal.endTime,
            proposal.executionTime,
            proposal.executed
        );
    }
    
    function getStakingInfo(address account) external view returns (
        uint256 staked,
        uint256 rewards,
        uint256 votingPowerAmount,
        uint256 stakingTime
    ) {
        return (
            stakedBalance[account],
            calculateRewards(account),
            votingPower[account],
            stakingTimestamp[account]
        );
    }
    
    function getTokenMetrics() external view returns (
        uint256 circulatingSupply,
        uint256 stakedSupply,
        uint256 burnedSupply,
        uint256 treasuryBalance,
        uint256 rewardsBalance
    ) {
        return (
            totalSupply(),
            totalStaked,
            totalBurned,
            _balances[treasury],
            _balances[stakingRewards]
        );
    }
}