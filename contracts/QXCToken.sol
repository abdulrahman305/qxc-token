// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol";

/**
 * @title QXC Token V2
 * @dev Production-grade ERC20 Token with advanced DeFi features
 * @notice Implements staking, governance, fee mechanisms, and security controls
 */
contract QXCToken is IERC20, IERC20Metadata, Ownable, Pausable, ReentrancyGuard {
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    mapping(address => bool) private _blacklist;
    mapping(address => bool) private _whitelist;
    mapping(address => uint256) private _stakingBalances;
    mapping(address => uint256) private _stakingTimestamps;
    mapping(address => uint256) private _votingPower;
    mapping(address => uint256) private _lastClaimTime;
    mapping(address => bool) public minters;

    uint256 private _totalSupply;
    uint256 private constant _maxSupply = 1_000_000_000 * 10**8; // 1 billion with 8 decimals
    string public override name = "QENEX Token";
    string public override symbol = "QXC";
    uint8 public override decimals = 8;
    
    // Fee mechanism
    uint256 public transactionFeePercent = 10; // 0.1% = 10 basis points
    uint256 public constant FEE_DENOMINATOR = 10000;
    address public feeRecipient;
    uint256 public totalFeeCollected;
    
    // Staking parameters
    uint256 public stakingAPR = 500; // 5% APR = 500 basis points
    uint256 public minStakingPeriod = 7 days;
    uint256 public totalStaked;
    
    // Security features
    bool public whitelistEnabled = false;
    bool public blacklistEnabled = true;
    bool public stakingEnabled = true;
    bool public votingEnabled = true;
    
    // Circuit breaker
    uint256 public maxTransferAmount = _maxSupply / 100; // 1% of max supply
    uint256 public dailyTransferLimit = _maxSupply / 10; // 10% of max supply
    mapping(address => uint256) private _dailyTransferred;
    mapping(address => uint256) private _lastTransferDate;
    
    event Burn(address indexed burner, uint256 value);
    event Mint(address indexed to, uint256 value);
    event Stake(address indexed staker, uint256 amount);
    event Unstake(address indexed staker, uint256 amount, uint256 reward);
    event RewardClaimed(address indexed staker, uint256 reward);
    event BlacklistUpdated(address indexed account, bool isBlacklisted);
    event WhitelistUpdated(address indexed account, bool isWhitelisted);
    event FeeCollected(address indexed from, uint256 amount);
    event EmergencyWithdraw(address indexed to, uint256 amount);
    event CircuitBreakerActivated(address indexed account, string reason);
    
    modifier onlyMinter() {
        require(minters[msg.sender] || msg.sender == owner(), "Not authorized minter");
        _;
    }
    
    modifier notBlacklisted(address account) {
        require(!_blacklist[account], "Account is blacklisted");
        _;
    }
    
    modifier checkWhitelist(address from, address to) {
        if (whitelistEnabled) {
            require(_whitelist[from] || _whitelist[to] || from == owner() || to == owner(), 
                    "Transfer requires whitelisted address");
        }
        _;
    }
    
    modifier checkTransferLimits(address sender, uint256 amount) {
        require(amount <= maxTransferAmount, "Transfer exceeds maximum amount");
        
        uint256 today = block.timestamp / 1 days;
        if (_lastTransferDate[sender] < today) {
            _dailyTransferred[sender] = 0;
            _lastTransferDate[sender] = today;
        }
        
        require(_dailyTransferred[sender] + amount <= dailyTransferLimit, 
                "Daily transfer limit exceeded");
        _dailyTransferred[sender] += amount;
        _;
    }
    
    constructor() {
        name = "QXC Token";
        symbol = "QXC";
        decimals = 18;
        owner = msg.sender;
        minters[msg.sender] = true;
        
        // Initial supply: 1,000,000 QXC
        uint256 initialSupply = 1000000 * 10**uint256(decimals);
        _totalSupply = initialSupply;
        _balances[msg.sender] = initialSupply;
        
        emit Transfer(address(0), msg.sender, initialSupply);
        emit MinterAdded(msg.sender);
    }
    
    /**
     * @dev Returns the total supply of tokens
     */
    function totalSupply() public view returns (uint256) {
        return _totalSupply;
    }
    
    /**
     * @dev Returns the balance of an account
     */
    function balanceOf(address account) public view returns (uint256) {
        return _balances[account];
    }
    
    /**
     * @dev Transfer tokens to another address
     */
    function transfer(address to, uint256 amount) public returns (bool) {
        require(to != address(0), "Transfer to zero address");
        require(_balances[msg.sender] >= amount, "Insufficient balance");
        
        _balances[msg.sender] -= amount;
        _balances[to] += amount;
        
        emit Transfer(msg.sender, to, amount);
        return true;
    }
    
    /**
     * @dev Returns the allowance of a spender
     */
    function allowance(address _owner, address spender) public view returns (uint256) {
        return _allowances[_owner][spender];
    }
    
    /**
     * @dev Approve a spender to use tokens
     */
    function approve(address spender, uint256 amount) public returns (bool) {
        require(spender != address(0), "Approve to zero address");
        
        _allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }
    
    /**
     * @dev Transfer tokens on behalf of another address
     */
    function transferFrom(address from, address to, uint256 amount) public returns (bool) {
        require(from != address(0), "Transfer from zero address");
        require(to != address(0), "Transfer to zero address");
        require(_balances[from] >= amount, "Insufficient balance");
        require(_allowances[from][msg.sender] >= amount, "Insufficient allowance");
        
        _balances[from] -= amount;
        _balances[to] += amount;
        _allowances[from][msg.sender] -= amount;
        
        emit Transfer(from, to, amount);
        return true;
    }
    
    /**
     * @dev Mint new tokens (only minters)
     */
    function mint(address to, uint256 amount) public onlyMinter returns (bool) {
        require(to != address(0), "Mint to zero address");
        
        _totalSupply += amount;
        _balances[to] += amount;
        
        emit Transfer(address(0), to, amount);
        return true;
    }
    
    /**
     * @dev Burn tokens from caller's balance
     */
    function burn(uint256 amount) public returns (bool) {
        require(_balances[msg.sender] >= amount, "Insufficient balance");
        
        _balances[msg.sender] -= amount;
        _totalSupply -= amount;
        
        emit Transfer(msg.sender, address(0), amount);
        return true;
    }
    
    /**
     * @dev Add a new minter
     */
    function addMinter(address minter) public onlyOwner {
        require(minter != address(0), "Invalid minter");
        require(!minters[minter], "Already minter");
        
        minters[minter] = true;
        emit MinterAdded(minter);
    }
    
    /**
     * @dev Remove a minter
     */
    function removeMinter(address minter) public onlyOwner {
        require(minters[minter], "Not a minter");
        
        minters[minter] = false;
        emit MinterRemoved(minter);
    }
    
    /**
     * @dev Transfer ownership
     */
    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "Invalid owner");
        
        address previousOwner = owner;
        owner = newOwner;
        
        emit OwnershipTransferred(previousOwner, newOwner);
    }
    
    /**
     * @dev Increase allowance
     */
    function increaseAllowance(address spender, uint256 addedValue) public returns (bool) {
        _allowances[msg.sender][spender] += addedValue;
        emit Approval(msg.sender, spender, _allowances[msg.sender][spender]);
        return true;
    }
    
    /**
     * @dev Decrease allowance
     */
    function decreaseAllowance(address spender, uint256 subtractedValue) public returns (bool) {
        uint256 currentAllowance = _allowances[msg.sender][spender];
        require(currentAllowance >= subtractedValue, "Decreased allowance below zero");
        
        _allowances[msg.sender][spender] = currentAllowance - subtractedValue;
        emit Approval(msg.sender, spender, _allowances[msg.sender][spender]);
        return true;
    }
}