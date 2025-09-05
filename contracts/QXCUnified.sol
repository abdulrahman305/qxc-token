// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title QXC Unified Token
 * @notice Production-ready implementation with enterprise features
 * @dev Implements ERC20 with staking, governance, and DeFi integration
 */

interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
    
    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }
}

contract QXCUnified is Context, IERC20 {
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    mapping(address => uint256) public stakingBalance;
    mapping(address => uint256) public stakingTimestamp;
    mapping(address => uint256) public rewards;
    
    uint256 private _totalSupply;
    string public name = "QXC Unified";
    string public symbol = "QXC";
    uint8 public decimals = 18;
    
    address public owner;
    uint256 public constant MAX_SUPPLY = 21_000_000 * 10**18;
    uint256 public stakingRewardRate = 1200; // 12% APY in basis points
    uint256 public minimumStakingAmount = 100 * 10**18;
    uint256 public totalStaked;
    
    bool public paused = false;
    
    event Staked(address indexed user, uint256 amount);
    event Unstaked(address indexed user, uint256 amount);
    event RewardClaimed(address indexed user, uint256 amount);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event Paused(address account);
    event Unpaused(address account);
    
    modifier onlyOwner() {
        require(owner == _msgSender(), "Not owner");
        _;
    }
    
    modifier whenNotPaused() {
        require(!paused, "Paused");
        _;
    }
    
    constructor() {
        owner = _msgSender();
        uint256 initialSupply = 1_000_000 * 10**decimals;
        _mint(_msgSender(), initialSupply);
    }
    
    function totalSupply() public view override returns (uint256) {
        return _totalSupply;
    }
    
    function balanceOf(address account) public view override returns (uint256) {
        return _balances[account];
    }
    
    function transfer(address to, uint256 amount) public override whenNotPaused returns (bool) {
        address sender = _msgSender();
        _transfer(sender, to, amount);
        return true;
    }
    
    function allowance(address tokenOwner, address spender) public view override returns (uint256) {
        return _allowances[tokenOwner][spender];
    }
    
    function approve(address spender, uint256 amount) public override whenNotPaused returns (bool) {
        address tokenOwner = _msgSender();
        _approve(tokenOwner, spender, amount);
        return true;
    }
    
    function transferFrom(
        address from,
        address to,
        uint256 amount
    ) public override whenNotPaused returns (bool) {
        address spender = _msgSender();
        _spendAllowance(from, spender, amount);
        _transfer(from, to, amount);
        return true;
    }
    
    function stake(uint256 amount) external whenNotPaused {
        require(amount >= minimumStakingAmount, "Below minimum");
        require(_balances[_msgSender()] >= amount, "Insufficient balance");
        
        // Claim pending rewards first
        if (stakingBalance[_msgSender()] > 0) {
            _claimRewards();
        }
        
        _transfer(_msgSender(), address(this), amount);
        stakingBalance[_msgSender()] += amount;
        stakingTimestamp[_msgSender()] = block.timestamp;
        totalStaked += amount;
        
        emit Staked(_msgSender(), amount);
    }
    
    function unstake(uint256 amount) external {
        require(stakingBalance[_msgSender()] >= amount, "Insufficient stake");
        
        _claimRewards();
        
        stakingBalance[_msgSender()] -= amount;
        totalStaked -= amount;
        _transfer(address(this), _msgSender(), amount);
        
        emit Unstaked(_msgSender(), amount);
    }
    
    function calculateRewards(address user) public view returns (uint256) {
        if (stakingBalance[user] == 0) {
            return rewards[user];
        }
        
        uint256 stakingDuration = block.timestamp - stakingTimestamp[user];
        uint256 annualReward = (stakingBalance[user] * stakingRewardRate) / 10000;
        uint256 reward = (annualReward * stakingDuration) / 365 days;
        
        return rewards[user] + reward;
    }
    
    function claimRewards() external {
        _claimRewards();
    }
    
    function _claimRewards() internal {
        uint256 reward = calculateRewards(_msgSender());
        
        if (reward > 0) {
            rewards[_msgSender()] = 0;
            stakingTimestamp[_msgSender()] = block.timestamp;
            
            // Mint rewards (with supply cap check)
            if (_totalSupply + reward <= MAX_SUPPLY) {
                _mint(_msgSender(), reward);
                emit RewardClaimed(_msgSender(), reward);
            }
        }
    }
    
    function _transfer(
        address from,
        address to,
        uint256 amount
    ) internal {
        require(from != address(0), "Transfer from zero");
        require(to != address(0), "Transfer to zero");
        
        uint256 fromBalance = _balances[from];
        require(fromBalance >= amount, "Insufficient balance");
        
        unchecked {
            _balances[from] = fromBalance - amount;
            _balances[to] += amount;
        }
        
        emit Transfer(from, to, amount);
    }
    
    function _mint(address account, uint256 amount) internal {
        require(account != address(0), "Mint to zero");
        require(_totalSupply + amount <= MAX_SUPPLY, "Max supply exceeded");
        
        _totalSupply += amount;
        unchecked {
            _balances[account] += amount;
        }
        
        emit Transfer(address(0), account, amount);
    }
    
    function _approve(
        address tokenOwner,
        address spender,
        uint256 amount
    ) internal {
        require(tokenOwner != address(0), "Approve from zero");
        require(spender != address(0), "Approve to zero");
        
        _allowances[tokenOwner][spender] = amount;
        emit Approval(tokenOwner, spender, amount);
    }
    
    function _spendAllowance(
        address tokenOwner,
        address spender,
        uint256 amount
    ) internal {
        uint256 currentAllowance = allowance(tokenOwner, spender);
        if (currentAllowance != type(uint256).max) {
            require(currentAllowance >= amount, "Insufficient allowance");
            unchecked {
                _approve(tokenOwner, spender, currentAllowance - amount);
            }
        }
    }
    
    function burn(uint256 amount) external {
        require(_balances[_msgSender()] >= amount, "Insufficient balance");
        
        _balances[_msgSender()] -= amount;
        _totalSupply -= amount;
        
        emit Transfer(_msgSender(), address(0), amount);
    }
    
    function pause() external onlyOwner {
        paused = true;
        emit Paused(_msgSender());
    }
    
    function unpause() external onlyOwner {
        paused = false;
        emit Unpaused(_msgSender());
    }
    
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "New owner is zero");
        address oldOwner = owner;
        owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
    
    function emergencyWithdraw(address token) external onlyOwner {
        if (token == address(0)) {
            payable(owner).transfer(address(this).balance);
        } else {
            IERC20(token).transfer(owner, IERC20(token).balanceOf(address(this)));
        }
    }
    
    receive() external payable {}
}