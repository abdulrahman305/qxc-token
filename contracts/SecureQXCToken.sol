// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title SecureQXCToken
 * @dev Production-ready ERC20 token with comprehensive security features
 * @notice Implements EIP-2612 permit, reentrancy guards, and MEV protection
 */

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Permit.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Snapshot.sol";
import "@openzeppelin/contracts/utils/math/SafeMath.sol";

contract SecureQXCToken is 
    ERC20Permit,
    ERC20Burnable,
    ERC20Snapshot,
    ReentrancyGuard,
    Pausable,
    AccessControl 
{
    using SafeMath for uint256;
    
    // Role definitions
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
    bytes32 public constant SNAPSHOT_ROLE = keccak256("SNAPSHOT_ROLE");
    bytes32 public constant BLACKLIST_ROLE = keccak256("BLACKLIST_ROLE");
    
    // Security features
    mapping(address => bool) public blacklisted;
    mapping(address => uint256) public lastTransferBlock;
    mapping(address => uint256) public dailyTransferLimit;
    mapping(address => uint256) public dailyTransferAmount;
    mapping(address => uint256) public lastTransferDay;
    
    // MEV Protection
    uint256 public constant MIN_DELAY_BLOCKS = 1;
    uint256 public constant MAX_TRANSFER_FREQUENCY = 1; // blocks
    
    // Fee mechanism
    uint256 public transferFeeRate = 30; // 0.3% = 30/10000
    uint256 public constant FEE_DENOMINATOR = 10000;
    address public feeRecipient;
    
    // Supply caps
    uint256 public constant MAX_SUPPLY = 1000000000 * 10**18; // 1 billion tokens
    uint256 public constant DAILY_MINT_LIMIT = 1000000 * 10**18; // 1 million tokens per day
    uint256 public dailyMintAmount;
    uint256 public lastMintDay;
    
    // Events
    event BlacklistUpdated(address indexed account, bool blacklisted);
    event FeeRateUpdated(uint256 oldRate, uint256 newRate);
    event FeeRecipientUpdated(address indexed oldRecipient, address indexed newRecipient);
    event DailyLimitUpdated(address indexed account, uint256 newLimit);
    event EmergencyWithdraw(address indexed recipient, uint256 amount);
    
    // Modifiers
    modifier notBlacklisted(address account) {
        require(!blacklisted[account], "Account is blacklisted");
        _;
    }
    
    modifier antiMEV(address from) {
        require(
            block.number >= lastTransferBlock[from].add(MAX_TRANSFER_FREQUENCY),
            "Transfer too frequent (MEV protection)"
        );
        _;
    }
    
    modifier withinDailyLimit(address from, uint256 amount) {
        uint256 today = block.timestamp / 1 days;
        
        if (lastTransferDay[from] < today) {
            dailyTransferAmount[from] = 0;
            lastTransferDay[from] = today;
        }
        
        if (dailyTransferLimit[from] > 0) {
            require(
                dailyTransferAmount[from].add(amount) <= dailyTransferLimit[from],
                "Daily transfer limit exceeded"
            );
        }
        _;
    }
    
    /**
     * @dev Constructor initializes token with security features
     */
    constructor() 
        ERC20("QENEX Token", "QXC") 
        ERC20Permit("QENEX Token") 
    {
        // Setup roles
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(MINTER_ROLE, msg.sender);
        _grantRole(PAUSER_ROLE, msg.sender);
        _grantRole(SNAPSHOT_ROLE, msg.sender);
        _grantRole(BLACKLIST_ROLE, msg.sender);
        
        // Set fee recipient
        feeRecipient = msg.sender;
        
        // Mint initial supply with cap check
        uint256 initialSupply = 1000000 * 10**18;
        require(initialSupply <= MAX_SUPPLY, "Exceeds max supply");
        _mint(msg.sender, initialSupply);
        
        // Initialize daily mint tracking
        lastMintDay = block.timestamp / 1 days;
    }
    
    /**
     * @dev Mint tokens with daily limit and supply cap
     */
    function mint(address to, uint256 amount) 
        public 
        onlyRole(MINTER_ROLE) 
        whenNotPaused
        nonReentrant
        notBlacklisted(to)
    {
        require(totalSupply().add(amount) <= MAX_SUPPLY, "Exceeds max supply");
        
        // Check daily mint limit
        uint256 today = block.timestamp / 1 days;
        if (lastMintDay < today) {
            dailyMintAmount = 0;
            lastMintDay = today;
        }
        
        require(
            dailyMintAmount.add(amount) <= DAILY_MINT_LIMIT,
            "Daily mint limit exceeded"
        );
        
        dailyMintAmount = dailyMintAmount.add(amount);
        _mint(to, amount);
    }
    
    /**
     * @dev Transfer with fees and security checks
     */
    function _transfer(
        address from,
        address to,
        uint256 amount
    ) 
        internal 
        virtual 
        override(ERC20, ERC20Snapshot)
        whenNotPaused
        notBlacklisted(from)
        notBlacklisted(to)
        antiMEV(from)
        withinDailyLimit(from, amount)
        nonReentrant
    {
        require(from != address(0), "Transfer from zero address");
        require(to != address(0), "Transfer to zero address");
        require(amount > 0, "Transfer amount must be greater than zero");
        
        // Update MEV protection
        lastTransferBlock[from] = block.number;
        
        // Update daily transfer amount
        uint256 today = block.timestamp / 1 days;
        if (lastTransferDay[from] < today) {
            dailyTransferAmount[from] = amount;
            lastTransferDay[from] = today;
        } else {
            dailyTransferAmount[from] = dailyTransferAmount[from].add(amount);
        }
        
        // Calculate fee
        uint256 fee = 0;
        if (transferFeeRate > 0 && feeRecipient != address(0)) {
            fee = amount.mul(transferFeeRate).div(FEE_DENOMINATOR);
            if (fee > 0) {
                super._transfer(from, feeRecipient, fee);
            }
        }
        
        // Transfer remaining amount
        uint256 transferAmount = amount.sub(fee);
        super._transfer(from, to, transferAmount);
    }
    
    /**
     * @dev Update blacklist status
     */
    function updateBlacklist(address account, bool _blacklisted) 
        external 
        onlyRole(BLACKLIST_ROLE) 
    {
        blacklisted[account] = _blacklisted;
        emit BlacklistUpdated(account, _blacklisted);
    }
    
    /**
     * @dev Update transfer fee rate
     */
    function updateFeeRate(uint256 newRate) 
        external 
        onlyRole(DEFAULT_ADMIN_ROLE) 
    {
        require(newRate <= 500, "Fee too high"); // Max 5%
        uint256 oldRate = transferFeeRate;
        transferFeeRate = newRate;
        emit FeeRateUpdated(oldRate, newRate);
    }
    
    /**
     * @dev Update fee recipient
     */
    function updateFeeRecipient(address newRecipient) 
        external 
        onlyRole(DEFAULT_ADMIN_ROLE) 
    {
        require(newRecipient != address(0), "Invalid recipient");
        address oldRecipient = feeRecipient;
        feeRecipient = newRecipient;
        emit FeeRecipientUpdated(oldRecipient, newRecipient);
    }
    
    /**
     * @dev Set daily transfer limit for an account
     */
    function setDailyTransferLimit(address account, uint256 limit) 
        external 
        onlyRole(DEFAULT_ADMIN_ROLE) 
    {
        dailyTransferLimit[account] = limit;
        emit DailyLimitUpdated(account, limit);
    }
    
    /**
     * @dev Pause token transfers
     */
    function pause() external onlyRole(PAUSER_ROLE) {
        _pause();
    }
    
    /**
     * @dev Unpause token transfers
     */
    function unpause() external onlyRole(PAUSER_ROLE) {
        _unpause();
    }
    
    /**
     * @dev Create snapshot of balances
     */
    function snapshot() external onlyRole(SNAPSHOT_ROLE) returns (uint256) {
        return _snapshot();
    }
    
    /**
     * @dev Emergency withdraw for stuck tokens
     */
    function emergencyWithdraw(address token, uint256 amount) 
        external 
        onlyRole(DEFAULT_ADMIN_ROLE)
        nonReentrant 
    {
        if (token == address(0)) {
            // Withdraw ETH
            (bool success, ) = msg.sender.call{value: amount}("");
            require(success, "ETH transfer failed");
        } else {
            // Withdraw ERC20 tokens
            IERC20(token).transfer(msg.sender, amount);
        }
        emit EmergencyWithdraw(token, amount);
    }
    
    /**
     * @dev Get current day for limit calculations
     */
    function getCurrentDay() external view returns (uint256) {
        return block.timestamp / 1 days;
    }
    
    /**
     * @dev Get remaining daily transfer amount for an account
     */
    function getRemainingDailyTransfer(address account) 
        external 
        view 
        returns (uint256) 
    {
        uint256 today = block.timestamp / 1 days;
        
        if (dailyTransferLimit[account] == 0) {
            return type(uint256).max;
        }
        
        if (lastTransferDay[account] < today) {
            return dailyTransferLimit[account];
        }
        
        if (dailyTransferAmount[account] >= dailyTransferLimit[account]) {
            return 0;
        }
        
        return dailyTransferLimit[account].sub(dailyTransferAmount[account]);
    }
    
    /**
     * @dev Get remaining daily mint amount
     */
    function getRemainingDailyMint() external view returns (uint256) {
        uint256 today = block.timestamp / 1 days;
        
        if (lastMintDay < today) {
            return DAILY_MINT_LIMIT;
        }
        
        if (dailyMintAmount >= DAILY_MINT_LIMIT) {
            return 0;
        }
        
        return DAILY_MINT_LIMIT.sub(dailyMintAmount);
    }
    
    /**
     * @dev Required overrides for multiple inheritance
     */
    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 amount
    ) internal override(ERC20, ERC20Snapshot) {
        super._beforeTokenTransfer(from, to, amount);
    }
    
    /**
     * @dev Receive function to accept ETH
     */
    receive() external payable {}
    
    /**
     * @dev Fallback function
     */
    fallback() external payable {}
}