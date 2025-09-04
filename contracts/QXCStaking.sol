// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title QXC Staking Contract
 * @dev Secure staking with proper rewards distribution and slashing
 */

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/math/SafeMath.sol";

contract QXCStaking is ReentrancyGuard, Pausable, Ownable {
    using SafeMath for uint256;
    using SafeERC20 for IERC20;
    
    IERC20 public immutable stakingToken;
    IERC20 public immutable rewardToken;
    
    // Staking parameters
    uint256 public constant MINIMUM_STAKE = 100 * 10**18;
    uint256 public constant LOCK_PERIOD = 7 days;
    uint256 public constant UNBONDING_PERIOD = 3 days;
    uint256 public constant MAX_VALIDATORS = 100;
    uint256 public constant SLASH_RATE = 1000; // 10% = 1000 / 10000
    
    // Reward parameters
    uint256 public rewardRate = 100; // Rewards per second
    uint256 public lastUpdateTime;
    uint256 public rewardPerTokenStored;
    uint256 public totalStaked;
    uint256 public totalRewards;
    
    // Validator info
    struct Validator {
        uint256 stakedAmount;
        uint256 rewardDebt;
        uint256 pendingRewards;
        uint256 lastStakeTime;
        uint256 unbondingAmount;
        uint256 unbondingTime;
        bool isActive;
        uint256 slashCount;
        uint256 performance; // 0-10000 basis points
    }
    
    mapping(address => Validator) public validators;
    address[] public validatorList;
    
    // Delegation info
    struct Delegation {
        uint256 amount;
        uint256 rewardDebt;
        uint256 lastDelegationTime;
    }
    
    mapping(address => mapping(address => Delegation)) public delegations;
    mapping(address => uint256) public totalDelegated;
    
    // Slashing info
    struct SlashingEvent {
        address validator;
        uint256 amount;
        uint256 timestamp;
        string reason;
    }
    
    SlashingEvent[] public slashingHistory;
    
    // Events
    event Staked(address indexed validator, uint256 amount);
    event Unstaked(address indexed validator, uint256 amount);
    event RewardsClaimed(address indexed validator, uint256 amount);
    event Delegated(address indexed delegator, address indexed validator, uint256 amount);
    event Undelegated(address indexed delegator, address indexed validator, uint256 amount);
    event Slashed(address indexed validator, uint256 amount, string reason);
    event RewardRateUpdated(uint256 newRate);
    event ValidatorActivated(address indexed validator);
    event ValidatorDeactivated(address indexed validator);
    
    // Modifiers
    modifier updateReward(address account) {
        rewardPerTokenStored = rewardPerToken();
        lastUpdateTime = block.timestamp;
        
        if (account != address(0)) {
            validators[account].pendingRewards = earned(account);
            validators[account].rewardDebt = rewardPerTokenStored;
        }
        _;
    }
    
    modifier onlyValidator() {
        require(validators[msg.sender].isActive, "Not an active validator");
        _;
    }
    
    constructor(address _stakingToken, address _rewardToken) {
        require(_stakingToken != address(0), "Invalid staking token");
        require(_rewardToken != address(0), "Invalid reward token");
        
        stakingToken = IERC20(_stakingToken);
        rewardToken = IERC20(_rewardToken);
        lastUpdateTime = block.timestamp;
    }
    
    // View functions
    function rewardPerToken() public view returns (uint256) {
        if (totalStaked == 0) {
            return rewardPerTokenStored;
        }
        
        return rewardPerTokenStored.add(
            block.timestamp.sub(lastUpdateTime).mul(rewardRate).mul(1e18).div(totalStaked)
        );
    }
    
    function earned(address account) public view returns (uint256) {
        Validator memory validator = validators[account];
        
        return validator.stakedAmount
            .mul(rewardPerToken().sub(validator.rewardDebt))
            .div(1e18)
            .add(validator.pendingRewards);
    }
    
    function getValidatorInfo(address validator) external view returns (
        uint256 stakedAmount,
        uint256 pendingRewards,
        uint256 performance,
        bool isActive
    ) {
        Validator memory v = validators[validator];
        return (
            v.stakedAmount,
            earned(validator),
            v.performance,
            v.isActive
        );
    }
    
    function getActiveValidators() external view returns (address[] memory) {
        uint256 count = 0;
        for (uint256 i = 0; i < validatorList.length; i++) {
            if (validators[validatorList[i]].isActive) {
                count++;
            }
        }
        
        address[] memory active = new address[](count);
        uint256 index = 0;
        for (uint256 i = 0; i < validatorList.length; i++) {
            if (validators[validatorList[i]].isActive) {
                active[index] = validatorList[i];
                index++;
            }
        }
        
        return active;
    }
    
    // Staking functions
    function stake(uint256 amount) 
        external 
        nonReentrant 
        whenNotPaused 
        updateReward(msg.sender) 
    {
        require(amount >= MINIMUM_STAKE, "Below minimum stake");
        require(validatorList.length < MAX_VALIDATORS, "Maximum validators reached");
        
        stakingToken.safeTransferFrom(msg.sender, address(this), amount);
        
        Validator storage validator = validators[msg.sender];
        
        if (!validator.isActive) {
            validatorList.push(msg.sender);
            validator.isActive = true;
            validator.performance = 10000; // Start at 100%
            emit ValidatorActivated(msg.sender);
        }
        
        validator.stakedAmount = validator.stakedAmount.add(amount);
        validator.lastStakeTime = block.timestamp;
        totalStaked = totalStaked.add(amount);
        
        emit Staked(msg.sender, amount);
    }
    
    function requestUnstake(uint256 amount) 
        external 
        nonReentrant 
        onlyValidator 
        updateReward(msg.sender) 
    {
        Validator storage validator = validators[msg.sender];
        require(validator.stakedAmount >= amount, "Insufficient stake");
        require(
            block.timestamp >= validator.lastStakeTime.add(LOCK_PERIOD),
            "Still in lock period"
        );
        require(validator.unbondingAmount == 0, "Unbonding in progress");
        
        validator.unbondingAmount = amount;
        validator.unbondingTime = block.timestamp.add(UNBONDING_PERIOD);
        validator.stakedAmount = validator.stakedAmount.sub(amount);
        totalStaked = totalStaked.sub(amount);
        
        // Deactivate if no stake left
        if (validator.stakedAmount == 0) {
            validator.isActive = false;
            emit ValidatorDeactivated(msg.sender);
        }
    }
    
    function completeUnstake() 
        external 
        nonReentrant 
        updateReward(msg.sender) 
    {
        Validator storage validator = validators[msg.sender];
        require(validator.unbondingAmount > 0, "No unbonding in progress");
        require(
            block.timestamp >= validator.unbondingTime,
            "Still in unbonding period"
        );
        
        uint256 amount = validator.unbondingAmount;
        validator.unbondingAmount = 0;
        validator.unbondingTime = 0;
        
        stakingToken.safeTransfer(msg.sender, amount);
        
        emit Unstaked(msg.sender, amount);
    }
    
    function claimRewards() 
        external 
        nonReentrant 
        updateReward(msg.sender) 
    {
        Validator storage validator = validators[msg.sender];
        uint256 reward = validator.pendingRewards;
        
        if (reward > 0) {
            validator.pendingRewards = 0;
            
            // Apply performance multiplier
            reward = reward.mul(validator.performance).div(10000);
            
            rewardToken.safeTransfer(msg.sender, reward);
            totalRewards = totalRewards.add(reward);
            
            emit RewardsClaimed(msg.sender, reward);
        }
    }
    
    // Delegation functions
    function delegate(address validator, uint256 amount) 
        external 
        nonReentrant 
        whenNotPaused 
    {
        require(validators[validator].isActive, "Validator not active");
        require(amount > 0, "Amount must be greater than 0");
        
        stakingToken.safeTransferFrom(msg.sender, address(this), amount);
        
        Delegation storage delegation = delegations[msg.sender][validator];
        delegation.amount = delegation.amount.add(amount);
        delegation.lastDelegationTime = block.timestamp;
        
        totalDelegated[validator] = totalDelegated[validator].add(amount);
        
        emit Delegated(msg.sender, validator, amount);
    }
    
    function undelegate(address validator, uint256 amount) 
        external 
        nonReentrant 
    {
        Delegation storage delegation = delegations[msg.sender][validator];
        require(delegation.amount >= amount, "Insufficient delegation");
        require(
            block.timestamp >= delegation.lastDelegationTime.add(LOCK_PERIOD),
            "Still in lock period"
        );
        
        delegation.amount = delegation.amount.sub(amount);
        totalDelegated[validator] = totalDelegated[validator].sub(amount);
        
        stakingToken.safeTransfer(msg.sender, amount);
        
        emit Undelegated(msg.sender, validator, amount);
    }
    
    // Slashing functions
    function slash(address validator, uint256 amount, string memory reason) 
        external 
        onlyOwner 
    {
        require(validators[validator].isActive, "Validator not active");
        require(amount <= validators[validator].stakedAmount.mul(SLASH_RATE).div(10000), "Slash amount too high");
        
        Validator storage v = validators[validator];
        v.stakedAmount = v.stakedAmount.sub(amount);
        v.slashCount = v.slashCount.add(1);
        
        // Reduce performance
        if (v.performance > 1000) {
            v.performance = v.performance.sub(1000); // Reduce by 10%
        } else {
            v.performance = 0;
        }
        
        totalStaked = totalStaked.sub(amount);
        
        // Record slashing event
        slashingHistory.push(SlashingEvent({
            validator: validator,
            amount: amount,
            timestamp: block.timestamp,
            reason: reason
        }));
        
        // Deactivate if slashed too many times
        if (v.slashCount >= 3) {
            v.isActive = false;
            emit ValidatorDeactivated(validator);
        }
        
        emit Slashed(validator, amount, reason);
    }
    
    // Admin functions
    function updateRewardRate(uint256 newRate) 
        external 
        onlyOwner 
        updateReward(address(0)) 
    {
        rewardRate = newRate;
        emit RewardRateUpdated(newRate);
    }
    
    function pause() external onlyOwner {
        _pause();
    }
    
    function unpause() external onlyOwner {
        _unpause();
    }
    
    function emergencyWithdraw() 
        external 
        nonReentrant 
    {
        Validator storage validator = validators[msg.sender];
        uint256 amount = validator.stakedAmount;
        
        if (amount > 0) {
            validator.stakedAmount = 0;
            validator.pendingRewards = 0;
            validator.isActive = false;
            totalStaked = totalStaked.sub(amount);
            
            // Apply penalty for emergency withdraw
            uint256 penalty = amount.mul(500).div(10000); // 5% penalty
            uint256 withdrawAmount = amount.sub(penalty);
            
            stakingToken.safeTransfer(msg.sender, withdrawAmount);
            
            emit Unstaked(msg.sender, withdrawAmount);
            emit ValidatorDeactivated(msg.sender);
        }
    }
    
    // Recovery function for stuck tokens
    function recoverERC20(address tokenAddress, uint256 amount) 
        external 
        onlyOwner 
    {
        require(
            tokenAddress != address(stakingToken) && tokenAddress != address(rewardToken),
            "Cannot recover staking or reward tokens"
        );
        IERC20(tokenAddress).safeTransfer(owner(), amount);
    }
}