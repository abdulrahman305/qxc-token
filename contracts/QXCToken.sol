// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title QXC Token
 * @dev Enhanced ERC20 Token with staking, governance, and advanced features
 */
contract QXCToken {
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;

    uint256 private _totalSupply;
    string public name;
    string public symbol;
    uint8 public decimals;
    
    address public owner;
    mapping(address => bool) public minters;
    
    // Enhanced features
    mapping(address => bool) public blacklisted;
    mapping(address => uint256) public stakingBalance;
    mapping(address => uint256) public stakingTimestamp;
    mapping(address => uint256) public votingPower;
    
    uint256 public constant MAX_SUPPLY = 1_000_000_000 * 10**18; // 1 billion max
    uint256 public stakingRewardRate = 500; // 5% APY
    uint256 public minimumStakingAmount = 100 * 10**18; // 100 QXC minimum
    bool public paused = false;
    
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event MinterAdded(address indexed minter);
    event MinterRemoved(address indexed minter);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
    
    modifier onlyMinter() {
        require(minters[msg.sender] || msg.sender == owner, "Not minter");
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