# QXC Token v3.0

🪙 **Enhanced ERC-20 Token** - QENEX native token with advanced DeFi features including staking, rewards, and governance.

## Features
- **ERC-20 Standard** - Full compatibility with Ethereum ecosystem
- **Staking System** - Earn rewards by staking tokens
- **Deflationary Mechanism** - Token burning capabilities
- **Governance Ready** - Built for DAO implementation
- **Security Features** - Pausable, access control, reentrancy protection

## Token Details
- **Name**: QENEX Token
- **Symbol**: QXC  
- **Decimals**: 8
- **Total Supply**: 1,000,000,000 QXC
- **Initial Supply**: 100,000,000 QXC

## Smart Contract Functions

### Staking
```solidity
function stake(uint256 amount) external
function unstake(uint256 amount) external  
function claimRewards() public
function calculateRewards(address user) public view returns (uint256)
```

### Token Management
```solidity
function mint(address to, uint256 amount) public onlyMinter
function burn(uint256 amount) public
function pause() public onlyOwner
```

## Deployment
```bash
npm install
npx hardhat compile
npx hardhat test
npx hardhat run scripts/deploy.js --network mainnet
```
