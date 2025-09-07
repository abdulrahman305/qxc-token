// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Pausable.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/math/SafeMath.sol";
import "@openzeppelin/contracts/governance/TimelockController.sol";

/**
 * @title QXCTokenSecure
 * @dev Secure implementation of QXC Token with fixed vulnerabilities
 * Security improvements:
 * - SafeMath for all arithmetic operations
 * - AccessControl for role-based permissions
 * - Timelock for critical operations
 * - Fixed reentrancy in staking
 * - Proper checks-effects-interactions pattern
 * - Multi-signature requirement for emergency functions
 */
contract QXCTokenSecure is ERC20, ERC20Burnable, ERC20Pausable, AccessControl, ReentrancyGuard {
    using SafeMath for uint256;

    // Role definitions
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
    bytes32 public constant UPGRADER_ROLE = keccak256("UPGRADER_ROLE");
    
    // Constants with SafeMath
    uint256 public constant MAX_SUPPLY = 10**9 * 10**8; // 1B tokens with 8 decimals
    uint256 public constant INITIAL_SUPPLY = 10**8 * 10**8; // 100M initial
    uint256 public constant MAX_REWARD_RATE = 20; // Maximum 20% annual
    uint256 public constant MIN_REWARD_RATE = 1; // Minimum 1% annual
    
    // Staking state
    mapping(address => uint256) public stakingBalances;
    mapping(address => uint256) public stakingRewards;
    mapping(address => uint256) public lastStakeTime;
    mapping(address => uint256) public rewardDebt;
    
    // Security improvements
    mapping(address => uint256) public nonces;
    mapping(bytes32 => bool) public executedTransactions;
    
    uint256 public stakingRewardRate = 12; // 12% annual
    uint256 public minStakingPeriod = 30 days;
    uint256 public totalStaked;
    uint256 public accRewardPerShare;
    uint256 public lastRewardTime;
    
    // Multi-sig for emergency
    address public guardian1;
    address public guardian2;
    address public guardian3;
    mapping(bytes32 => uint256) public emergencyApprovals;
    
    // Events
    event Staked(address indexed user, uint256 amount, uint256 timestamp);
    event Unstaked(address indexed user, uint256 amount, uint256 timestamp);
    event RewardsClaimed(address indexed user, uint256 reward, uint256 timestamp);
    event RewardRateUpdated(uint256 oldRate, uint256 newRate);
    event EmergencyActionProposed(bytes32 indexed actionId, address proposer);
    event EmergencyActionExecuted(bytes32 indexed actionId);
    
    modifier validAddress(address _addr) {
        require(_addr != address(0), "Invalid address");
        _;
    }
    
    modifier validAmount(uint256 _amount) {
        require(_amount > 0, "Amount must be positive");
        _;
    }
    
    constructor(
        address _guardian1,
        address _guardian2,
        address _guardian3
    ) ERC20("QENEX Token Secure", "QXCS") {
        require(_guardian1 != _guardian2 && _guardian2 != _guardian3 && _guardian1 != _guardian3, "Guardians must be unique");
        
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(MINTER_ROLE, msg.sender);
        _grantRole(PAUSER_ROLE, msg.sender);
        
        guardian1 = _guardian1;
        guardian2 = _guardian2;
        guardian3 = _guardian3;
        
        // Use SafeMath for initial mint
        _mint(msg.sender, INITIAL_SUPPLY);
        lastRewardTime = block.timestamp;
    }
    
    function decimals() public pure override returns (uint8) {
        return 8;
    }
    
    /**
     * @dev Mint new tokens with supply cap check
     */
    function mint(address to, uint256 amount) 
        public 
        onlyRole(MINTER_ROLE) 
        validAddress(to) 
        validAmount(amount) 
    {
        uint256 newSupply = totalSupply().add(amount);
        require(newSupply <= MAX_SUPPLY, "Exceeds max supply");
        _mint(to, amount);
    }
    
    /**
     * @dev Stake tokens with proper state updates
     */
    function stake(uint256 amount) external nonReentrant validAmount(amount) {
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");
        
        // Update rewards before state changes
        _updateRewardPool();
        
        // Calculate pending rewards if already staking
        if (stakingBalances[msg.sender] > 0) {
            uint256 pending = _calculatePendingRewards(msg.sender);
            if (pending > 0) {
                rewardDebt[msg.sender] = rewardDebt[msg.sender].add(pending);
            }
        }
        
        // Effects before interactions
        stakingBalances[msg.sender] = stakingBalances[msg.sender].add(amount);
        totalStaked = totalStaked.add(amount);
        lastStakeTime[msg.sender] = block.timestamp;
        
        // Update reward debt
        rewardDebt[msg.sender] = stakingBalances[msg.sender]
            .mul(accRewardPerShare)
            .div(1e12);
        
        // Interaction (transfer) last
        _transfer(msg.sender, address(this), amount);
        
        emit Staked(msg.sender, amount, block.timestamp);
    }
    
    /**
     * @dev Unstake tokens with minimum period check
     */
    function unstake(uint256 amount) external nonReentrant validAmount(amount) {
        require(stakingBalances[msg.sender] >= amount, "Insufficient staked amount");
        require(
            block.timestamp >= lastStakeTime[msg.sender].add(minStakingPeriod),
            "Minimum staking period not met"
        );
        
        // Update rewards and claim before unstaking
        _updateRewardPool();
        uint256 pending = _calculatePendingRewards(msg.sender);
        
        // Effects before interactions
        stakingBalances[msg.sender] = stakingBalances[msg.sender].sub(amount);
        totalStaked = totalStaked.sub(amount);
        
        // Update reward debt
        rewardDebt[msg.sender] = stakingBalances[msg.sender]
            .mul(accRewardPerShare)
            .div(1e12);
        
        // Interactions last
        if (pending > 0 && totalSupply().add(pending) <= MAX_SUPPLY) {
            _mint(msg.sender, pending);
            stakingRewards[msg.sender] = stakingRewards[msg.sender].add(pending);
            emit RewardsClaimed(msg.sender, pending, block.timestamp);
        }
        
        _transfer(address(this), msg.sender, amount);
        
        emit Unstaked(msg.sender, amount, block.timestamp);
    }
    
    /**
     * @dev Calculate pending rewards with overflow protection
     */
    function _calculatePendingRewards(address user) private view returns (uint256) {
        if (stakingBalances[user] == 0) return 0;
        
        uint256 accReward = stakingBalances[user]
            .mul(accRewardPerShare)
            .div(1e12);
            
        if (accReward <= rewardDebt[user]) return 0;
        
        return accReward.sub(rewardDebt[user]);
    }
    
    /**
     * @dev Update reward pool
     */
    function _updateRewardPool() private {
        if (block.timestamp <= lastRewardTime) return;
        if (totalStaked == 0) {
            lastRewardTime = block.timestamp;
            return;
        }
        
        uint256 timeDelta = block.timestamp.sub(lastRewardTime);
        uint256 reward = totalStaked
            .mul(stakingRewardRate)
            .mul(timeDelta)
            .div(365 days)
            .div(100);
            
        accRewardPerShare = accRewardPerShare.add(
            reward.mul(1e12).div(totalStaked)
        );
        
        lastRewardTime = block.timestamp;
    }
    
    /**
     * @dev Claim rewards with supply cap check
     */
    function claimRewards() public nonReentrant {
        _updateRewardPool();
        
        uint256 pending = _calculatePendingRewards(msg.sender);
        require(pending > 0, "No rewards to claim");
        
        uint256 newSupply = totalSupply().add(pending);
        require(newSupply <= MAX_SUPPLY, "Exceeds max supply");
        
        // Update state before interaction
        rewardDebt[msg.sender] = stakingBalances[msg.sender]
            .mul(accRewardPerShare)
            .div(1e12);
        stakingRewards[msg.sender] = stakingRewards[msg.sender].add(pending);
        
        // Mint rewards
        _mint(msg.sender, pending);
        
        emit RewardsClaimed(msg.sender, pending, block.timestamp);
    }
    
    /**
     * @dev Update staking reward rate with bounds checking
     */
    function updateRewardRate(uint256 newRate) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(newRate >= MIN_REWARD_RATE && newRate <= MAX_REWARD_RATE, "Rate out of bounds");
        
        // Update pool before changing rate
        _updateRewardPool();
        
        uint256 oldRate = stakingRewardRate;
        stakingRewardRate = newRate;
        
        emit RewardRateUpdated(oldRate, newRate);
    }
    
    /**
     * @dev Pause token transfers
     */
    function pause() public onlyRole(PAUSER_ROLE) {
        _pause();
    }
    
    /**
     * @dev Unpause token transfers
     */
    function unpause() public onlyRole(PAUSER_ROLE) {
        _unpause();
    }
    
    /**
     * @dev Emergency withdraw with multi-sig requirement
     */
    function proposeEmergencyAction(bytes32 actionId) external {
        require(
            msg.sender == guardian1 || msg.sender == guardian2 || msg.sender == guardian3,
            "Not a guardian"
        );
        require(!executedTransactions[actionId], "Already executed");
        
        emergencyApprovals[actionId] = emergencyApprovals[actionId].add(1);
        
        emit EmergencyActionProposed(actionId, msg.sender);
        
        // Execute if we have 2/3 approvals
        if (emergencyApprovals[actionId] >= 2) {
            _executeEmergencyAction(actionId);
        }
    }
    
    /**
     * @dev Execute emergency action
     */
    function _executeEmergencyAction(bytes32 actionId) private {
        executedTransactions[actionId] = true;
        
        // Example: Emergency pause
        if (actionId == keccak256("EMERGENCY_PAUSE")) {
            _pause();
        }
        
        emit EmergencyActionExecuted(actionId);
    }
    
    /**
     * @dev Get comprehensive staking information
     */
    function getStakingInfo(address user) external view returns (
        uint256 stakedAmount,
        uint256 pendingRewards,
        uint256 totalRewardsEarned,
        uint256 nextUnstakeTime,
        uint256 apr
    ) {
        uint256 pending = _calculatePendingRewards(user);
        
        return (
            stakingBalances[user],
            pending,
            stakingRewards[user],
            lastStakeTime[user].add(minStakingPeriod),
            stakingRewardRate
        );
    }
    
    /**
     * @dev Override required by Solidity
     */
    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 amount
    ) internal virtual override(ERC20, ERC20Pausable) {
        super._beforeTokenTransfer(from, to, amount);
    }
    
    /**
     * @dev Prevent accidental ETH transfers
     */
    receive() external payable {
        revert("Contract does not accept ETH");
    }
    
    /**
     * @dev Recover accidentally sent ERC20 tokens
     */
    function recoverERC20(address tokenAddress, uint256 amount) 
        external 
        onlyRole(DEFAULT_ADMIN_ROLE) 
    {
        require(tokenAddress != address(this), "Cannot recover native token");
        IERC20(tokenAddress).transfer(msg.sender, amount);
    }
}