// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Pausable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

contract QXCToken is ERC20, ERC20Burnable, ERC20Pausable, Ownable, ReentrancyGuard {
    uint256 public constant MAX_SUPPLY = 1000000000 * 10**8; // 1B tokens with 8 decimals
    uint256 public constant INITIAL_SUPPLY = 100000000 * 10**8; // 100M initial
    
    mapping(address => bool) public minters;
    mapping(address => uint256) public stakingBalances;
    mapping(address => uint256) public stakingRewards;
    mapping(address => uint256) public lastStakeTime;
    
    uint256 public stakingRewardRate = 12; // 12% annual
    uint256 public minStakingPeriod = 30 days;
    
    event Staked(address indexed user, uint256 amount);
    event Unstaked(address indexed user, uint256 amount);
    event RewardsClaimed(address indexed user, uint256 reward);
    event MinterAdded(address indexed minter);
    event MinterRemoved(address indexed minter);
    
    modifier onlyMinter() {
        require(minters[msg.sender], "Not a minter");
        _;
    }
    
    constructor() ERC20("QENEX Token", "QXC") {
        _mint(msg.sender, INITIAL_SUPPLY);
        minters[msg.sender] = true;
    }
    
    function decimals() public pure override returns (uint8) {
        return 8;
    }
    
    function mint(address to, uint256 amount) public onlyMinter {
        require(totalSupply() + amount <= MAX_SUPPLY, "Exceeds max supply");
        _mint(to, amount);
    }
    
    function addMinter(address minter) external onlyOwner {
        minters[minter] = true;
        emit MinterAdded(minter);
    }
    
    function removeMinter(address minter) external onlyOwner {
        minters[minter] = false;
        emit MinterRemoved(minter);
    }
    
    function stake(uint256 amount) external nonReentrant {
        require(amount > 0, "Amount must be greater than 0");
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");
        
        // Claim existing rewards before staking more
        if (stakingBalances[msg.sender] > 0) {
            claimRewards();
        }
        
        _transfer(msg.sender, address(this), amount);
        stakingBalances[msg.sender] += amount;
        lastStakeTime[msg.sender] = block.timestamp;
        
        emit Staked(msg.sender, amount);
    }
    
    function unstake(uint256 amount) external nonReentrant {
        require(stakingBalances[msg.sender] >= amount, "Insufficient staked amount");
        require(
            block.timestamp >= lastStakeTime[msg.sender] + minStakingPeriod,
            "Minimum staking period not met"
        );
        
        // Claim rewards before unstaking
        claimRewards();
        
        stakingBalances[msg.sender] -= amount;
        _transfer(address(this), msg.sender, amount);
        
        emit Unstaked(msg.sender, amount);
    }
    
    function calculateRewards(address user) public view returns (uint256) {
        if (stakingBalances[user] == 0) return 0;
        
        uint256 stakingDuration = block.timestamp - lastStakeTime[user];
        uint256 annualReward = (stakingBalances[user] * stakingRewardRate) / 100;
        uint256 reward = (annualReward * stakingDuration) / (365 days);
        
        return reward;
    }
    
    function claimRewards() public nonReentrant {
        uint256 reward = calculateRewards(msg.sender);
        if (reward > 0 && totalSupply() + reward <= MAX_SUPPLY) {
            stakingRewards[msg.sender] += reward;
            lastStakeTime[msg.sender] = block.timestamp;
            _mint(msg.sender, reward);
            
            emit RewardsClaimed(msg.sender, reward);
        }
    }
    
    function pause() public onlyOwner {
        _pause();
    }
    
    function unpause() public onlyOwner {
        _unpause();
    }
    
    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 amount
    ) internal virtual override(ERC20, ERC20Pausable) {
        super._beforeTokenTransfer(from, to, amount);
    }
    
    // Emergency functions
    function emergencyWithdraw() external onlyOwner {
        uint256 balance = address(this).balance;
        if (balance > 0) {
            payable(owner()).transfer(balance);
        }
    }
    
    function getStakingInfo(address user) external view returns (
        uint256 stakedAmount,
        uint256 pendingRewards,
        uint256 totalRewardsEarned,
        uint256 nextUnstakeTime
    ) {
        return (
            stakingBalances[user],
            calculateRewards(user),
            stakingRewards[user],
            lastStakeTime[user] + minStakingPeriod
        );
    }
}
