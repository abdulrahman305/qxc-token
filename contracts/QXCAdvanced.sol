// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

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

abstract contract Ownable is Context {
    address private _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor() {
        _transferOwnership(_msgSender());
    }

    modifier onlyOwner() {
        require(owner() == _msgSender(), "Ownable: caller is not the owner");
        _;
    }

    function owner() public view virtual returns (address) {
        return _owner;
    }

    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        _transferOwnership(newOwner);
    }

    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

abstract contract Pausable is Context {
    event Paused(address account);
    event Unpaused(address account);

    bool private _paused;

    constructor() {
        _paused = false;
    }

    modifier whenNotPaused() {
        require(!paused(), "Pausable: paused");
        _;
    }

    modifier whenPaused() {
        require(paused(), "Pausable: not paused");
        _;
    }

    function paused() public view virtual returns (bool) {
        return _paused;
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

contract QXCAdvanced is Context, IERC20, IERC20Metadata, Ownable, Pausable {
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    
    mapping(address => bool) public blacklisted;
    mapping(address => uint256) public stakingBalance;
    mapping(address => uint256) public stakingTimestamp;
    mapping(address => uint256) public rewards;
    mapping(address => bool) public minters;
    
    uint256 private _totalSupply;
    string public name;
    string public symbol;
    uint8 public decimals;
    
    uint256 public constant MAX_SUPPLY = 21_000_000 * 10**18;
    uint256 public stakingRewardRate = 1200; // 12% APY in basis points
    uint256 public minimumStakingAmount = 100 * 10**18;
    uint256 public totalStaked;
    uint256 public rewardPool;
    
    uint256 public transactionFee = 10; // 0.1% in basis points
    address public feeCollector;
    uint256 public collectedFees;
    
    mapping(address => uint256) public votingPower;
    mapping(uint256 => Proposal) public proposals;
    uint256 public proposalCount;
    
    struct Proposal {
        string description;
        uint256 forVotes;
        uint256 againstVotes;
        uint256 deadline;
        bool executed;
        mapping(address => bool) hasVoted;
    }
    
    event Staked(address indexed user, uint256 amount);
    event Unstaked(address indexed user, uint256 amount);
    event RewardClaimed(address indexed user, uint256 amount);
    event ProposalCreated(uint256 indexed proposalId, string description);
    event Voted(uint256 indexed proposalId, address indexed voter, bool support, uint256 weight);
    event ProposalExecuted(uint256 indexed proposalId);
    event FeeCollected(address indexed from, uint256 amount);
    event Blacklisted(address indexed account);
    event Whitelisted(address indexed account);
    event MinterAdded(address indexed account);
    event MinterRemoved(address indexed account);
    
    constructor() {
        name = "QXC Advanced Token";
        symbol = "QXC";
        decimals = 18;
        
        uint256 initialSupply = 1_000_000 * 10**decimals;
        _totalSupply = initialSupply;
        _balances[_msgSender()] = initialSupply;
        
        rewardPool = 1_000_000 * 10**decimals;
        feeCollector = _msgSender();
        minters[_msgSender()] = true;
        
        emit Transfer(address(0), _msgSender(), initialSupply);
    }
    
    modifier notBlacklisted(address account) {
        require(!blacklisted[account], "Account is blacklisted");
        _;
    }
    
    modifier onlyMinter() {
        require(minters[_msgSender()], "Not a minter");
        _;
    }
    
    function totalSupply() public view virtual override returns (uint256) {
        return _totalSupply;
    }
    
    function balanceOf(address account) public view virtual override returns (uint256) {
        return _balances[account];
    }
    
    function transfer(address to, uint256 amount) 
        public 
        virtual 
        override 
        whenNotPaused
        notBlacklisted(_msgSender())
        notBlacklisted(to)
        returns (bool) 
    {
        address owner = _msgSender();
        _transferWithFee(owner, to, amount);
        return true;
    }
    
    function allowance(address owner, address spender) public view virtual override returns (uint256) {
        return _allowances[owner][spender];
    }
    
    function approve(address spender, uint256 amount) 
        public 
        virtual 
        override
        whenNotPaused 
        returns (bool) 
    {
        address owner = _msgSender();
        _approve(owner, spender, amount);
        return true;
    }
    
    function transferFrom(address from, address to, uint256 amount) 
        public 
        virtual 
        override
        whenNotPaused
        notBlacklisted(from)
        notBlacklisted(to)
        returns (bool) 
    {
        address spender = _msgSender();
        _spendAllowance(from, spender, amount);
        _transferWithFee(from, to, amount);
        return true;
    }
    
    function _transferWithFee(address from, address to, uint256 amount) internal {
        uint256 fee = (amount * transactionFee) / 10000;
        uint256 netAmount = amount - fee;
        
        _transfer(from, to, netAmount);
        
        if (fee > 0) {
            _transfer(from, feeCollector, fee);
            collectedFees += fee;
            emit FeeCollected(from, fee);
        }
    }
    
    function _transfer(address from, address to, uint256 amount) internal virtual {
        require(from != address(0), "ERC20: transfer from the zero address");
        require(to != address(0), "ERC20: transfer to the zero address");
        
        uint256 fromBalance = _balances[from];
        require(fromBalance >= amount, "ERC20: transfer amount exceeds balance");
        
        unchecked {
            _balances[from] = fromBalance - amount;
            _balances[to] += amount;
        }
        
        emit Transfer(from, to, amount);
    }
    
    function _approve(address owner, address spender, uint256 amount) internal virtual {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        
        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }
    
    function _spendAllowance(address owner, address spender, uint256 amount) internal virtual {
        uint256 currentAllowance = allowance(owner, spender);
        if (currentAllowance != type(uint256).max) {
            require(currentAllowance >= amount, "ERC20: insufficient allowance");
            unchecked {
                _approve(owner, spender, currentAllowance - amount);
            }
        }
    }
    
    function stake(uint256 amount) external whenNotPaused notBlacklisted(_msgSender()) {
        require(amount >= minimumStakingAmount, "Amount below minimum");
        require(_balances[_msgSender()] >= amount, "Insufficient balance");
        
        if (stakingBalance[_msgSender()] > 0) {
            _claimRewards();
        }
        
        _transfer(_msgSender(), address(this), amount);
        
        stakingBalance[_msgSender()] += amount;
        stakingTimestamp[_msgSender()] = block.timestamp;
        totalStaked += amount;
        
        _updateVotingPower(_msgSender());
        
        emit Staked(_msgSender(), amount);
    }
    
    function unstake(uint256 amount) external whenNotPaused {
        require(stakingBalance[_msgSender()] >= amount, "Insufficient staked balance");
        require(block.timestamp >= stakingTimestamp[_msgSender()] + 7 days, "Tokens locked");
        
        _claimRewards();
        
        stakingBalance[_msgSender()] -= amount;
        totalStaked -= amount;
        
        _transfer(address(this), _msgSender(), amount);
        
        _updateVotingPower(_msgSender());
        
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
    
    function claimRewards() external whenNotPaused {
        _claimRewards();
    }
    
    function _claimRewards() internal {
        uint256 reward = calculateRewards(_msgSender());
        
        if (reward > 0 && reward <= rewardPool) {
            rewards[_msgSender()] = 0;
            stakingTimestamp[_msgSender()] = block.timestamp;
            
            rewardPool -= reward;
            _balances[_msgSender()] += reward;
            _totalSupply += reward;
            
            emit RewardClaimed(_msgSender(), reward);
            emit Transfer(address(0), _msgSender(), reward);
        }
    }
    
    function _updateVotingPower(address user) internal {
        votingPower[user] = stakingBalance[user] + (_balances[user] / 2);
    }
    
    function createProposal(string memory description) external returns (uint256) {
        require(votingPower[_msgSender()] >= 1000 * 10**decimals, "Insufficient voting power");
        
        proposalCount++;
        Proposal storage proposal = proposals[proposalCount];
        proposal.description = description;
        proposal.deadline = block.timestamp + 3 days;
        
        emit ProposalCreated(proposalCount, description);
        
        return proposalCount;
    }
    
    function vote(uint256 proposalId, bool support) external {
        Proposal storage proposal = proposals[proposalId];
        require(block.timestamp < proposal.deadline, "Voting ended");
        require(!proposal.hasVoted[_msgSender()], "Already voted");
        require(votingPower[_msgSender()] > 0, "No voting power");
        
        proposal.hasVoted[_msgSender()] = true;
        
        if (support) {
            proposal.forVotes += votingPower[_msgSender()];
        } else {
            proposal.againstVotes += votingPower[_msgSender()];
        }
        
        emit Voted(proposalId, _msgSender(), support, votingPower[_msgSender()]);
    }
    
    function executeProposal(uint256 proposalId) external {
        Proposal storage proposal = proposals[proposalId];
        require(block.timestamp >= proposal.deadline, "Voting not ended");
        require(!proposal.executed, "Already executed");
        require(proposal.forVotes > proposal.againstVotes, "Proposal rejected");
        
        proposal.executed = true;
        
        emit ProposalExecuted(proposalId);
    }
    
    function mint(address to, uint256 amount) external onlyMinter {
        require(_totalSupply + amount <= MAX_SUPPLY, "Max supply exceeded");
        
        _totalSupply += amount;
        _balances[to] += amount;
        
        emit Transfer(address(0), to, amount);
    }
    
    function burn(uint256 amount) external {
        require(_balances[_msgSender()] >= amount, "Insufficient balance");
        
        _balances[_msgSender()] -= amount;
        _totalSupply -= amount;
        
        emit Transfer(_msgSender(), address(0), amount);
    }
    
    function addMinter(address account) external onlyOwner {
        minters[account] = true;
        emit MinterAdded(account);
    }
    
    function removeMinter(address account) external onlyOwner {
        minters[account] = false;
        emit MinterRemoved(account);
    }
    
    function blacklistAccount(address account) external onlyOwner {
        blacklisted[account] = true;
        emit Blacklisted(account);
    }
    
    function whitelistAccount(address account) external onlyOwner {
        blacklisted[account] = false;
        emit Whitelisted(account);
    }
    
    function setTransactionFee(uint256 fee) external onlyOwner {
        require(fee <= 100, "Fee too high"); // Max 1%
        transactionFee = fee;
    }
    
    function setFeeCollector(address collector) external onlyOwner {
        require(collector != address(0), "Invalid address");
        feeCollector = collector;
    }
    
    function setStakingRewardRate(uint256 rate) external onlyOwner {
        require(rate <= 10000, "Rate too high");
        stakingRewardRate = rate;
    }
    
    function addRewardPool(uint256 amount) external onlyOwner {
        require(_balances[_msgSender()] >= amount, "Insufficient balance");
        
        _transfer(_msgSender(), address(this), amount);
        rewardPool += amount;
    }
    
    function pause() external onlyOwner {
        _pause();
    }
    
    function unpause() external onlyOwner {
        _unpause();
    }
    
    function emergencyWithdraw() external onlyOwner {
        uint256 balance = _balances[address(this)];
        if (balance > 0) {
            _transfer(address(this), owner(), balance);
        }
    }
    
    receive() external payable {
        revert("Direct ETH transfers not accepted");
    }
    
    fallback() external payable {
        revert("Fallback not allowed");
    }
}