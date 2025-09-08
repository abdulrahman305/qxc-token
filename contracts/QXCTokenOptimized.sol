// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Pausable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title QXCTokenOptimized
 * @dev Gas-optimized version of QXCToken with improved performance
 */
contract QXCTokenOptimized is ERC20, ERC20Burnable, ERC20Pausable, Ownable, ReentrancyGuard {
    // Constants - cheaper than storage variables
    uint256 private constant MAX_SUPPLY = 1000000000 * 10**8; // 1B tokens with 8 decimals
    uint256 private constant INITIAL_SUPPLY = 100000000 * 10**8; // 100M initial
    uint256 private constant STAKING_REWARD_RATE = 12; // 12% annual
    uint256 private constant MIN_STAKING_PERIOD = 30 days;
    uint256 private constant SECONDS_PER_YEAR = 365 days;
    
    // Packed struct to save gas (uses single storage slot)
    struct StakeInfo {
        uint128 balance;      // Staked balance (128 bits is enough for most cases)
        uint64 lastStakeTime; // Timestamp (64 bits covers timestamps until year 584942417355)
        uint64 rewards;       // Accumulated rewards
    }
    
    // Single mapping instead of multiple - saves gas
    mapping(address => StakeInfo) public stakes;
    mapping(address => bool) public minters;
    
    // Batch operations support
    uint256 private constant MAX_BATCH_SIZE = 100;
    
    // Events
    event Staked(address indexed user, uint256 amount);
    event Unstaked(address indexed user, uint256 amount);
    event RewardsClaimed(address indexed user, uint256 reward);
    event MinterAdded(address indexed minter);
    event MinterRemoved(address indexed minter);
    event BatchTransfer(address indexed from, uint256 totalAmount, uint256 recipients);
    
    // Custom errors (cheaper than require strings)
    error NotMinter();
    error ExceedsMaxSupply();
    error InvalidAmount();
    error InsufficientBalance();
    error InsufficientStakedAmount();
    error StakingPeriodNotMet();
    error BatchSizeTooLarge();
    error ArrayLengthMismatch();
    
    modifier onlyMinter() {
        if (!minters[msg.sender]) revert NotMinter();
        _;
    }
    
    constructor() ERC20("QENEX Token", "QXC") {
        _mint(msg.sender, INITIAL_SUPPLY);
        minters[msg.sender] = true;
    }
    
    function decimals() public pure override returns (uint8) {
        return 8;
    }
    
    /**
     * @dev Optimized mint function with custom error
     */
    function mint(address to, uint256 amount) public onlyMinter {
        if (totalSupply() + amount > MAX_SUPPLY) revert ExceedsMaxSupply();
        _mint(to, amount);
    }
    
    /**
     * @dev Batch mint to multiple addresses - saves gas on multiple transactions
     */
    function batchMint(address[] calldata recipients, uint256[] calldata amounts) 
        external 
        onlyMinter 
    {
        uint256 length = recipients.length;
        if (length != amounts.length) revert ArrayLengthMismatch();
        if (length > MAX_BATCH_SIZE) revert BatchSizeTooLarge();
        
        uint256 totalAmount;
        for (uint256 i; i < length;) {
            totalAmount += amounts[i];
            unchecked { ++i; }
        }
        
        if (totalSupply() + totalAmount > MAX_SUPPLY) revert ExceedsMaxSupply();
        
        for (uint256 i; i < length;) {
            _mint(recipients[i], amounts[i]);
            unchecked { ++i; }
        }
    }
    
    /**
     * @dev Batch transfer - saves gas for multiple transfers
     */
    function batchTransfer(address[] calldata recipients, uint256[] calldata amounts) 
        external 
        nonReentrant 
        returns (bool) 
    {
        uint256 length = recipients.length;
        if (length != amounts.length) revert ArrayLengthMismatch();
        if (length > MAX_BATCH_SIZE) revert BatchSizeTooLarge();
        
        uint256 totalAmount;
        for (uint256 i; i < length;) {
            totalAmount += amounts[i];
            unchecked { ++i; }
        }
        
        if (balanceOf(msg.sender) < totalAmount) revert InsufficientBalance();
        
        for (uint256 i; i < length;) {
            _transfer(msg.sender, recipients[i], amounts[i]);
            unchecked { ++i; }
        }
        
        emit BatchTransfer(msg.sender, totalAmount, length);
        return true;
    }
    
    function addMinter(address minter) external onlyOwner {
        minters[minter] = true;
        emit MinterAdded(minter);
    }
    
    function removeMinter(address minter) external onlyOwner {
        minters[minter] = false;
        emit MinterRemoved(minter);
    }
    
    /**
     * @dev Optimized stake function using packed struct
     */
    function stake(uint256 amount) external nonReentrant {
        if (amount == 0) revert InvalidAmount();
        if (balanceOf(msg.sender) < amount) revert InsufficientBalance();
        
        StakeInfo storage info = stakes[msg.sender];
        
        // Auto-claim existing rewards if any
        if (info.balance > 0) {
            _claimRewardsInternal(msg.sender);
        }
        
        _transfer(msg.sender, address(this), amount);
        
        // Safe casting with overflow check
        info.balance = uint128(uint256(info.balance) + amount);
        info.lastStakeTime = uint64(block.timestamp);
        
        emit Staked(msg.sender, amount);
    }
    
    /**
     * @dev Batch stake for multiple users (admin function)
     */
    function batchStakeFor(
        address[] calldata users, 
        uint256[] calldata amounts
    ) external onlyOwner {
        uint256 length = users.length;
        if (length != amounts.length) revert ArrayLengthMismatch();
        if (length > MAX_BATCH_SIZE) revert BatchSizeTooLarge();
        
        for (uint256 i; i < length;) {
            address user = users[i];
            uint256 amount = amounts[i];
            
            if (amount > 0 && balanceOf(user) >= amount) {
                StakeInfo storage info = stakes[user];
                
                _transfer(user, address(this), amount);
                info.balance = uint128(uint256(info.balance) + amount);
                info.lastStakeTime = uint64(block.timestamp);
                
                emit Staked(user, amount);
            }
            
            unchecked { ++i; }
        }
    }
    
    /**
     * @dev Optimized unstake function
     */
    function unstake(uint256 amount) external nonReentrant {
        StakeInfo storage info = stakes[msg.sender];
        
        if (info.balance < amount) revert InsufficientStakedAmount();
        if (block.timestamp < info.lastStakeTime + MIN_STAKING_PERIOD) {
            revert StakingPeriodNotMet();
        }
        
        // Auto-claim rewards before unstaking
        _claimRewardsInternal(msg.sender);
        
        info.balance = uint128(uint256(info.balance) - amount);
        _transfer(address(this), msg.sender, amount);
        
        emit Unstaked(msg.sender, amount);
    }
    
    /**
     * @dev Calculate rewards with optimized math
     */
    function calculateRewards(address user) public view returns (uint256) {
        StakeInfo memory info = stakes[user];
        if (info.balance == 0) return 0;
        
        uint256 stakingDuration = block.timestamp - info.lastStakeTime;
        
        // Optimized calculation to prevent overflow and save gas
        // reward = (balance * rate * duration) / (100 * SECONDS_PER_YEAR)
        uint256 reward = (uint256(info.balance) * STAKING_REWARD_RATE * stakingDuration) 
                        / (100 * SECONDS_PER_YEAR);
        
        return reward;
    }
    
    /**
     * @dev Internal function for claiming rewards
     */
    function _claimRewardsInternal(address user) private {
        uint256 reward = calculateRewards(user);
        if (reward > 0 && totalSupply() + reward <= MAX_SUPPLY) {
            StakeInfo storage info = stakes[user];
            info.rewards = uint64(uint256(info.rewards) + reward);
            info.lastStakeTime = uint64(block.timestamp);
            _mint(user, reward);
            
            emit RewardsClaimed(user, reward);
        }
    }
    
    /**
     * @dev Public claim rewards function
     */
    function claimRewards() external nonReentrant {
        _claimRewardsInternal(msg.sender);
    }
    
    /**
     * @dev Batch claim rewards for multiple users
     */
    function batchClaimRewards(address[] calldata users) external {
        uint256 length = users.length;
        if (length > MAX_BATCH_SIZE) revert BatchSizeTooLarge();
        
        for (uint256 i; i < length;) {
            _claimRewardsInternal(users[i]);
            unchecked { ++i; }
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
    
    /**
     * @dev Emergency withdraw with gas optimization
     */
    function emergencyWithdraw() external onlyOwner {
        uint256 balance = address(this).balance;
        if (balance > 0) {
            (bool success, ) = payable(owner()).call{value: balance}("");
            require(success, "Transfer failed");
        }
    }
    
    /**
     * @dev Get staking info with single read
     */
    function getStakingInfo(address user) external view returns (
        uint256 stakedAmount,
        uint256 pendingRewards,
        uint256 totalRewardsEarned,
        uint256 nextUnstakeTime
    ) {
        StakeInfo memory info = stakes[user];
        return (
            info.balance,
            calculateRewards(user),
            info.rewards,
            info.lastStakeTime + MIN_STAKING_PERIOD
        );
    }
    
    /**
     * @dev Batch get staking info - reduces RPC calls
     */
    function batchGetStakingInfo(address[] calldata users) 
        external 
        view 
        returns (
            uint256[] memory stakedAmounts,
            uint256[] memory pendingRewards
        ) 
    {
        uint256 length = users.length;
        stakedAmounts = new uint256[](length);
        pendingRewards = new uint256[](length);
        
        for (uint256 i; i < length;) {
            StakeInfo memory info = stakes[users[i]];
            stakedAmounts[i] = info.balance;
            pendingRewards[i] = calculateRewards(users[i]);
            unchecked { ++i; }
        }
    }
}