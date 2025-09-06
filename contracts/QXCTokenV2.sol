// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Snapshot.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Permit.sol";

/**
 * @title QXCTokenV2
 * @dev Enhanced QENEX token with banking features and DeFi integration
 */
contract QXCTokenV2 is ERC20, ERC20Burnable, ERC20Snapshot, AccessControl, Pausable, ERC20Permit {
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant SNAPSHOT_ROLE = keccak256("SNAPSHOT_ROLE");
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
    bytes32 public constant COMPLIANCE_ROLE = keccak256("COMPLIANCE_ROLE");
    
    uint256 public constant MAX_SUPPLY = 1_000_000_000 * 10**18; // 1 billion tokens
    uint256 public constant INITIAL_SUPPLY = 100_000_000 * 10**18; // 100 million initial
    
    mapping(address => bool) public blacklisted;
    mapping(address => bool) public whitelisted;
    mapping(address => uint256) public vestingSchedule;
    mapping(address => uint256) public stakingBalance;
    mapping(address => uint256) public stakingTimestamp;
    
    bool public whitelistEnabled = false;
    uint256 public minStakingPeriod = 30 days;
    uint256 public stakingAPY = 500; // 5% in basis points
    
    event Blacklisted(address indexed account);
    event Whitelisted(address indexed account);
    event Staked(address indexed user, uint256 amount);
    event Unstaked(address indexed user, uint256 amount, uint256 reward);
    event VestingScheduleCreated(address indexed beneficiary, uint256 amount, uint256 releaseTime);
    
    constructor() 
        ERC20("QENEX Token", "QXC") 
        ERC20Permit("QENEX Token")
    {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(MINTER_ROLE, msg.sender);
        _grantRole(SNAPSHOT_ROLE, msg.sender);
        _grantRole(PAUSER_ROLE, msg.sender);
        _grantRole(COMPLIANCE_ROLE, msg.sender);
        
        _mint(msg.sender, INITIAL_SUPPLY);
    }
    
    /**
     * @dev Mint new tokens (controlled supply)
     */
    function mint(address to, uint256 amount) public onlyRole(MINTER_ROLE) {
        require(totalSupply() + amount <= MAX_SUPPLY, "Exceeds max supply");
        _mint(to, amount);
    }
    
    /**
     * @dev Create snapshot for governance
     */
    function snapshot() public onlyRole(SNAPSHOT_ROLE) returns (uint256) {
        return _snapshot();
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
     * @dev Add address to blacklist
     */
    function addToBlacklist(address account) public onlyRole(COMPLIANCE_ROLE) {
        blacklisted[account] = true;
        emit Blacklisted(account);
    }
    
    /**
     * @dev Remove address from blacklist
     */
    function removeFromBlacklist(address account) public onlyRole(COMPLIANCE_ROLE) {
        blacklisted[account] = false;
    }
    
    /**
     * @dev Add address to whitelist
     */
    function addToWhitelist(address account) public onlyRole(COMPLIANCE_ROLE) {
        whitelisted[account] = true;
        emit Whitelisted(account);
    }
    
    /**
     * @dev Remove address from whitelist
     */
    function removeFromWhitelist(address account) public onlyRole(COMPLIANCE_ROLE) {
        whitelisted[account] = false;
    }
    
    /**
     * @dev Enable/disable whitelist requirement
     */
    function setWhitelistEnabled(bool enabled) public onlyRole(DEFAULT_ADMIN_ROLE) {
        whitelistEnabled = enabled;
    }
    
    /**
     * @dev Create vesting schedule for tokens
     */
    function createVestingSchedule(
        address beneficiary,
        uint256 amount,
        uint256 releaseTime
    ) public onlyRole(DEFAULT_ADMIN_ROLE) {
        require(releaseTime > block.timestamp, "Release time in past");
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");
        
        _transfer(msg.sender, address(this), amount);
        vestingSchedule[beneficiary] = releaseTime;
        
        emit VestingScheduleCreated(beneficiary, amount, releaseTime);
    }
    
    /**
     * @dev Release vested tokens
     */
    function releaseVestedTokens() public {
        require(vestingSchedule[msg.sender] > 0, "No vesting schedule");
        require(block.timestamp >= vestingSchedule[msg.sender], "Tokens not vested yet");
        
        uint256 amount = balanceOf(address(this));
        vestingSchedule[msg.sender] = 0;
        _transfer(address(this), msg.sender, amount);
    }
    
    /**
     * @dev Stake tokens for rewards
     */
    function stake(uint256 amount) public whenNotPaused {
        require(amount > 0, "Cannot stake 0");
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");
        
        // Calculate and distribute pending rewards if already staking
        if (stakingBalance[msg.sender] > 0) {
            uint256 reward = calculateStakingReward(msg.sender);
            if (reward > 0) {
                _mint(msg.sender, reward);
            }
        }
        
        _transfer(msg.sender, address(this), amount);
        stakingBalance[msg.sender] += amount;
        stakingTimestamp[msg.sender] = block.timestamp;
        
        emit Staked(msg.sender, amount);
    }
    
    /**
     * @dev Unstake tokens and claim rewards
     */
    function unstake() public {
        require(stakingBalance[msg.sender] > 0, "No staked balance");
        require(
            block.timestamp >= stakingTimestamp[msg.sender] + minStakingPeriod,
            "Min staking period not met"
        );
        
        uint256 stakedAmount = stakingBalance[msg.sender];
        uint256 reward = calculateStakingReward(msg.sender);
        
        stakingBalance[msg.sender] = 0;
        stakingTimestamp[msg.sender] = 0;
        
        _transfer(address(this), msg.sender, stakedAmount);
        if (reward > 0) {
            _mint(msg.sender, reward);
        }
        
        emit Unstaked(msg.sender, stakedAmount, reward);
    }
    
    /**
     * @dev Calculate staking rewards
     */
    function calculateStakingReward(address user) public view returns (uint256) {
        if (stakingBalance[user] == 0) {
            return 0;
        }
        
        uint256 stakingDuration = block.timestamp - stakingTimestamp[user];
        uint256 annualReward = (stakingBalance[user] * stakingAPY) / 10000;
        uint256 reward = (annualReward * stakingDuration) / 365 days;
        
        return reward;
    }
    
    /**
     * @dev Update staking APY
     */
    function updateStakingAPY(uint256 newAPY) public onlyRole(DEFAULT_ADMIN_ROLE) {
        require(newAPY <= 2000, "APY too high"); // Max 20%
        stakingAPY = newAPY;
    }
    
    /**
     * @dev Update minimum staking period
     */
    function updateMinStakingPeriod(uint256 newPeriod) public onlyRole(DEFAULT_ADMIN_ROLE) {
        minStakingPeriod = newPeriod;
    }
    
    /**
     * @dev Get staking info for user
     */
    function getStakingInfo(address user) public view returns (
        uint256 balance,
        uint256 timestamp,
        uint256 pendingReward
    ) {
        return (
            stakingBalance[user],
            stakingTimestamp[user],
            calculateStakingReward(user)
        );
    }
    
    /**
     * @dev Override _beforeTokenTransfer to add compliance checks
     */
    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 amount
    ) internal override(ERC20, ERC20Snapshot) whenNotPaused {
        super._beforeTokenTransfer(from, to, amount);
        
        require(!blacklisted[from], "Sender blacklisted");
        require(!blacklisted[to], "Recipient blacklisted");
        
        if (whitelistEnabled) {
            require(whitelisted[from] || from == address(0), "Sender not whitelisted");
            require(whitelisted[to] || to == address(0), "Recipient not whitelisted");
        }
    }
    
    /**
     * @dev Batch transfer for efficiency
     */
    function batchTransfer(
        address[] calldata recipients,
        uint256[] calldata amounts
    ) public whenNotPaused {
        require(recipients.length == amounts.length, "Arrays length mismatch");
        require(recipients.length <= 100, "Too many recipients");
        
        for (uint256 i = 0; i < recipients.length; i++) {
            _transfer(msg.sender, recipients[i], amounts[i]);
        }
    }
    
    /**
     * @dev Rescue stuck tokens
     */
    function rescueTokens(
        address token,
        address to,
        uint256 amount
    ) public onlyRole(DEFAULT_ADMIN_ROLE) {
        if (token == address(0)) {
            payable(to).transfer(amount);
        } else {
            IERC20(token).transfer(to, amount);
        }
    }
    
    /**
     * @dev Get circulating supply (total - locked in contract)
     */
    function circulatingSupply() public view returns (uint256) {
        return totalSupply() - balanceOf(address(this));
    }
    
    /**
     * @dev Get total value locked in staking
     */
    function totalValueLocked() public view returns (uint256) {
        return balanceOf(address(this));
    }
}