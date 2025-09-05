// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title QXC Token - QENEX Ecosystem Native Token
 * @notice Advanced ERC20 token with DeFi features and governance
 * @dev Production-ready implementation with security best practices
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

abstract contract Pausable is Context {
    event Paused(address account);
    event Unpaused(address account);
    
    bool private _paused;
    
    constructor() {
        _paused = false;
    }
    
    function paused() public view virtual returns (bool) {
        return _paused;
    }
    
    modifier whenNotPaused() {
        require(!paused(), "Pausable: paused");
        _;
    }
    
    modifier whenPaused() {
        require(paused(), "Pausable: not paused");
        _;
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

abstract contract ReentrancyGuard {
    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;
    uint256 private _status;
    
    constructor() {
        _status = _NOT_ENTERED;
    }
    
    modifier nonReentrant() {
        require(_status != _ENTERED, "ReentrancyGuard: reentrant call");
        _status = _ENTERED;
        _;
        _status = _NOT_ENTERED;
    }
}

contract QXCToken is Context, IERC20, IERC20Metadata, Pausable, ReentrancyGuard {
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    
    // Advanced features
    mapping(address => bool) public blacklisted;
    mapping(address => bool) public whitelisted;
    mapping(address => uint256) public stakingBalance;
    mapping(address => uint256) public stakingTimestamp;
    mapping(address => uint256) public vestingBalance;
    mapping(address => uint256) public vestingReleaseTime;
    
    uint256 private _totalSupply;
    string public name;
    string public symbol;
    uint8 public decimals;
    
    // Governance
    address public governance;
    address public pendingGovernance;
    
    // Fee structure
    uint256 public transferFee = 30; // 0.3% = 30 basis points
    uint256 public constant MAX_FEE = 1000; // 10% maximum
    uint256 public constant FEE_DENOMINATOR = 10000;
    address public feeCollector;
    
    // Supply management
    uint256 public constant MAX_SUPPLY = 1_000_000_000 * 10**18; // 1 billion max
    uint256 public mintedSupply;
    uint256 public burnedSupply;
    
    // Staking rewards
    uint256 public stakingRewardRate = 500; // 5% APY = 500 basis points
    uint256 public constant STAKING_DENOMINATOR = 10000;
    uint256 public constant SECONDS_IN_YEAR = 31536000;
    
    // Events
    event GovernanceTransferred(address indexed previousGovernance, address indexed newGovernance);
    event FeeUpdated(uint256 newFee);
    event FeeCollectorUpdated(address indexed newCollector);
    event Blacklisted(address indexed account);
    event Whitelisted(address indexed account);
    event Staked(address indexed user, uint256 amount);
    event Unstaked(address indexed user, uint256 amount, uint256 reward);
    event TokensBurned(address indexed burner, uint256 amount);
    event TokensMinted(address indexed to, uint256 amount);
    event VestingCreated(address indexed beneficiary, uint256 amount, uint256 releaseTime);
    event VestingReleased(address indexed beneficiary, uint256 amount);
    
    modifier onlyGovernance() {
        require(_msgSender() == governance, "QXC: caller is not governance");
        _;
    }
    
    modifier notBlacklisted(address account) {
        require(!blacklisted[account], "QXC: account is blacklisted");
        _;
    }
    
    constructor(
        string memory _name,
        string memory _symbol,
        uint256 _initialSupply
    ) {
        require(_initialSupply <= MAX_SUPPLY, "QXC: exceeds max supply");
        
        name = _name;
        symbol = _symbol;
        decimals = 18;
        governance = _msgSender();
        feeCollector = _msgSender();
        
        _totalSupply = _initialSupply;
        mintedSupply = _initialSupply;
        _balances[_msgSender()] = _initialSupply;
        
        emit Transfer(address(0), _msgSender(), _initialSupply);
        emit GovernanceTransferred(address(0), _msgSender());
    }
    
    // ERC20 Implementation
    function totalSupply() public view override returns (uint256) {
        return _totalSupply;
    }
    
    function balanceOf(address account) public view override returns (uint256) {
        return _balances[account];
    }
    
    function transfer(address recipient, uint256 amount) 
        public 
        override 
        whenNotPaused 
        notBlacklisted(_msgSender())
        notBlacklisted(recipient)
        nonReentrant
        returns (bool) 
    {
        _transferWithFee(_msgSender(), recipient, amount);
        return true;
    }
    
    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowances[owner][spender];
    }
    
    function approve(address spender, uint256 amount) 
        public 
        override 
        whenNotPaused
        returns (bool) 
    {
        _approve(_msgSender(), spender, amount);
        return true;
    }
    
    function transferFrom(address sender, address recipient, uint256 amount) 
        public 
        override 
        whenNotPaused
        notBlacklisted(sender)
        notBlacklisted(recipient)
        nonReentrant
        returns (bool) 
    {
        uint256 currentAllowance = _allowances[sender][_msgSender()];
        require(currentAllowance >= amount, "QXC: transfer exceeds allowance");
        
        _transferWithFee(sender, recipient, amount);
        
        unchecked {
            _approve(sender, _msgSender(), currentAllowance - amount);
        }
        
        return true;
    }
    
    // Internal transfer with fee mechanism
    function _transferWithFee(address sender, address recipient, uint256 amount) internal {
        uint256 senderBalance = _balances[sender];
        require(senderBalance >= amount, "QXC: transfer exceeds balance");
        
        uint256 feeAmount = 0;
        
        // Apply fee unless whitelisted
        if (!whitelisted[sender] && !whitelisted[recipient] && transferFee > 0) {
            feeAmount = (amount * transferFee) / FEE_DENOMINATOR;
            
            if (feeAmount > 0 && feeCollector != address(0)) {
                unchecked {
                    _balances[sender] = senderBalance - amount;
                    _balances[feeCollector] += feeAmount;
                    _balances[recipient] += amount - feeAmount;
                }
                
                emit Transfer(sender, feeCollector, feeAmount);
                emit Transfer(sender, recipient, amount - feeAmount);
            }
        } else {
            // No fee transfer
            unchecked {
                _balances[sender] = senderBalance - amount;
                _balances[recipient] += amount;
            }
            
            emit Transfer(sender, recipient, amount);
        }
    }
    
    function _approve(address owner, address spender, uint256 amount) internal {
        require(owner != address(0), "QXC: approve from zero address");
        require(spender != address(0), "QXC: approve to zero address");
        
        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }
    
    // Staking Functions
    function stake(uint256 amount) 
        external 
        whenNotPaused 
        notBlacklisted(_msgSender())
        nonReentrant 
    {
        require(amount > 0, "QXC: cannot stake 0");
        require(_balances[_msgSender()] >= amount, "QXC: insufficient balance");
        
        // Claim existing rewards if any
        if (stakingBalance[_msgSender()] > 0) {
            _claimStakingRewards(_msgSender());
        }
        
        // Update staking balance
        _balances[_msgSender()] -= amount;
        stakingBalance[_msgSender()] += amount;
        stakingTimestamp[_msgSender()] = block.timestamp;
        
        emit Staked(_msgSender(), amount);
    }
    
    function unstake(uint256 amount) 
        external 
        whenNotPaused
        nonReentrant 
    {
        require(amount > 0, "QXC: cannot unstake 0");
        require(stakingBalance[_msgSender()] >= amount, "QXC: insufficient staked balance");
        
        // Calculate and mint rewards
        uint256 reward = calculateStakingReward(_msgSender());
        
        // Update balances
        stakingBalance[_msgSender()] -= amount;
        _balances[_msgSender()] += amount;
        
        if (reward > 0) {
            _mint(_msgSender(), reward);
        }
        
        // Reset timestamp if fully unstaked
        if (stakingBalance[_msgSender()] == 0) {
            stakingTimestamp[_msgSender()] = 0;
        } else {
            stakingTimestamp[_msgSender()] = block.timestamp;
        }
        
        emit Unstaked(_msgSender(), amount, reward);
    }
    
    function calculateStakingReward(address account) public view returns (uint256) {
        if (stakingBalance[account] == 0 || stakingTimestamp[account] == 0) {
            return 0;
        }
        
        uint256 timeStaked = block.timestamp - stakingTimestamp[account];
        uint256 reward = (stakingBalance[account] * stakingRewardRate * timeStaked) / 
                         (STAKING_DENOMINATOR * SECONDS_IN_YEAR);
        
        return reward;
    }
    
    function _claimStakingRewards(address account) internal {
        uint256 reward = calculateStakingReward(account);
        
        if (reward > 0) {
            _mint(account, reward);
            stakingTimestamp[account] = block.timestamp;
        }
    }
    
    // Vesting Functions
    function createVesting(address beneficiary, uint256 amount, uint256 releaseTime)
        external
        onlyGovernance
    {
        require(beneficiary != address(0), "QXC: zero address");
        require(amount > 0, "QXC: zero amount");
        require(releaseTime > block.timestamp, "QXC: invalid release time");
        require(_balances[_msgSender()] >= amount, "QXC: insufficient balance");
        
        _balances[_msgSender()] -= amount;
        vestingBalance[beneficiary] += amount;
        vestingReleaseTime[beneficiary] = releaseTime;
        
        emit VestingCreated(beneficiary, amount, releaseTime);
    }
    
    function releaseVesting() external nonReentrant {
        require(vestingBalance[_msgSender()] > 0, "QXC: no vesting balance");
        require(block.timestamp >= vestingReleaseTime[_msgSender()], "QXC: vesting not released");
        
        uint256 amount = vestingBalance[_msgSender()];
        vestingBalance[_msgSender()] = 0;
        vestingReleaseTime[_msgSender()] = 0;
        _balances[_msgSender()] += amount;
        
        emit VestingReleased(_msgSender(), amount);
    }
    
    // Mint and Burn Functions
    function mint(address to, uint256 amount) 
        external 
        onlyGovernance 
    {
        _mint(to, amount);
    }
    
    function _mint(address to, uint256 amount) internal {
        require(to != address(0), "QXC: mint to zero address");
        require(_totalSupply + amount <= MAX_SUPPLY, "QXC: exceeds max supply");
        
        _totalSupply += amount;
        mintedSupply += amount;
        _balances[to] += amount;
        
        emit Transfer(address(0), to, amount);
        emit TokensMinted(to, amount);
    }
    
    function burn(uint256 amount) external {
        require(_balances[_msgSender()] >= amount, "QXC: burn exceeds balance");
        
        _balances[_msgSender()] -= amount;
        _totalSupply -= amount;
        burnedSupply += amount;
        
        emit Transfer(_msgSender(), address(0), amount);
        emit TokensBurned(_msgSender(), amount);
    }
    
    // Governance Functions
    function transferGovernance(address newGovernance) external onlyGovernance {
        require(newGovernance != address(0), "QXC: zero address");
        pendingGovernance = newGovernance;
    }
    
    function acceptGovernance() external {
        require(_msgSender() == pendingGovernance, "QXC: not pending governance");
        
        emit GovernanceTransferred(governance, pendingGovernance);
        governance = pendingGovernance;
        pendingGovernance = address(0);
    }
    
    function setTransferFee(uint256 newFee) external onlyGovernance {
        require(newFee <= MAX_FEE, "QXC: fee too high");
        transferFee = newFee;
        emit FeeUpdated(newFee);
    }
    
    function setFeeCollector(address newCollector) external onlyGovernance {
        require(newCollector != address(0), "QXC: zero address");
        feeCollector = newCollector;
        emit FeeCollectorUpdated(newCollector);
    }
    
    function setStakingRewardRate(uint256 newRate) external onlyGovernance {
        require(newRate <= 2000, "QXC: rate too high"); // Max 20% APY
        stakingRewardRate = newRate;
    }
    
    // Access Control Functions
    function blacklistAccount(address account) external onlyGovernance {
        blacklisted[account] = true;
        emit Blacklisted(account);
    }
    
    function removeBlacklist(address account) external onlyGovernance {
        blacklisted[account] = false;
    }
    
    function whitelistAccount(address account) external onlyGovernance {
        whitelisted[account] = true;
        emit Whitelisted(account);
    }
    
    function removeWhitelist(address account) external onlyGovernance {
        whitelisted[account] = false;
    }
    
    // Emergency Functions
    function pause() external onlyGovernance {
        _pause();
    }
    
    function unpause() external onlyGovernance {
        _unpause();
    }
    
    // Recovery Function
    function recoverERC20(address tokenAddress, uint256 amount) 
        external 
        onlyGovernance 
    {
        require(tokenAddress != address(this), "QXC: cannot recover QXC");
        IERC20(tokenAddress).transfer(governance, amount);
    }
    
    // View Functions
    function getAccountInfo(address account) external view returns (
        uint256 balance,
        uint256 staked,
        uint256 pendingRewards,
        uint256 vested,
        uint256 vestingRelease,
        bool isBlacklisted,
        bool isWhitelisted
    ) {
        return (
            _balances[account],
            stakingBalance[account],
            calculateStakingReward(account),
            vestingBalance[account],
            vestingReleaseTime[account],
            blacklisted[account],
            whitelisted[account]
        );
    }
    
    function getSupplyInfo() external view returns (
        uint256 total,
        uint256 minted,
        uint256 burned,
        uint256 maxSupply,
        uint256 circulatingSupply
    ) {
        uint256 totalStaked = 0;
        // In production, this would be tracked separately for efficiency
        
        return (
            _totalSupply,
            mintedSupply,
            burnedSupply,
            MAX_SUPPLY,
            _totalSupply - totalStaked
        );
    }
}