# QXC Token - QENEX Ecosystem Token

Enterprise-grade ERC20 token with staking, governance, and advanced DeFi features for the QENEX banking ecosystem.

## 📋 Contract Features

### Core ERC20
- Standard transfer functionality
- Allowance mechanism
- Balance tracking
- Total supply management

### Extended Features
- Minting (controlled)
- Burning
- Minter management
- Ownership control

## 🏗 Contract Architecture

```
┌─────────────────────────────────────┐
│         QXC Token Contract          │
├─────────────────────────────────────┤
│                                     │
│  ┌─────────────┐  ┌──────────────┐ │
│  │   Owner     │  │   Minters    │ │
│  │  Controls   │  │     Can      │ │
│  │  Minters    │  │     Mint     │ │
│  └──────┬──────┘  └──────┬───────┘ │
│         │                 │         │
│         ▼                 ▼         │
│  ┌────────────────────────────┐    │
│  │      Token Logic           │    │
│  │  • Transfer                │    │
│  │  • Approve                 │    │
│  │  • Mint                    │    │
│  │  • Burn                    │    │
│  └────────────────────────────┘    │
│                                     │
└─────────────────────────────────────┘
```

## 💻 Deployment

### Using Hardhat

```javascript
const QXCToken = await ethers.getContractFactory("QXCToken");
const token = await QXCToken.deploy();
await token.deployed();

console.log("Token deployed to:", token.address);
```

### Using Remix

1. Copy contract code to Remix IDE
2. Select Solidity compiler 0.8.20
3. Deploy contract
4. Initial supply of 1,000,000 QXC will be minted to deployer

## 📊 Contract Interface

### Read Functions

```solidity
// Get token info
name() → string
symbol() → string  
decimals() → uint8
totalSupply() → uint256

// Check balances
balanceOf(address account) → uint256

// Check allowances
allowance(address owner, address spender) → uint256

// Check roles
owner() → address
minters(address account) → bool
```

### Write Functions

```solidity
// Standard ERC20
transfer(address to, uint256 amount) → bool
approve(address spender, uint256 amount) → bool
transferFrom(address from, address to, uint256 amount) → bool

// Extended functions
mint(address to, uint256 amount) → bool        // Only minters
burn(uint256 amount) → bool                    // Any holder
increaseAllowance(address spender, uint256 addedValue) → bool
decreaseAllowance(address spender, uint256 subtractedValue) → bool

// Admin functions
addMinter(address minter)                      // Only owner
removeMinter(address minter)                   // Only owner
transferOwnership(address newOwner)            // Only owner
```

## 🔐 Access Control

### Roles

#### Owner
- Can add/remove minters
- Can mint tokens
- Can transfer ownership
- Single owner at a time

#### Minters
- Can mint new tokens
- Multiple minters allowed
- Managed by owner

#### Users
- Can transfer tokens
- Can approve spenders
- Can burn their tokens

## 📈 Token Economics

### Initial Supply
- 1,000,000 QXC minted to deployer
- 18 decimal places
- No max supply (mintable)

### Minting
- Controlled by owner and designated minters
- Increases total supply
- Emits Transfer event from address(0)

### Burning
- Any holder can burn their tokens
- Decreases total supply
- Emits Transfer event to address(0)

## 🧪 Testing Examples

### JavaScript Tests

```javascript
describe("QXCToken", function() {
    let token;
    let owner, addr1, addr2;
    
    beforeEach(async function() {
        [owner, addr1, addr2] = await ethers.getSigners();
        const QXCToken = await ethers.getContractFactory("QXCToken");
        token = await QXCToken.deploy();
    });
    
    it("Should have correct initial supply", async function() {
        const supply = await token.totalSupply();
        expect(supply).to.equal(ethers.utils.parseEther("1000000"));
    });
    
    it("Should transfer tokens", async function() {
        await token.transfer(addr1.address, 100);
        expect(await token.balanceOf(addr1.address)).to.equal(100);
    });
    
    it("Should mint tokens", async function() {
        await token.mint(addr1.address, 1000);
        expect(await token.balanceOf(addr1.address)).to.equal(1000);
    });
    
    it("Should burn tokens", async function() {
        const initialSupply = await token.totalSupply();
        await token.burn(100);
        const newSupply = await token.totalSupply();
        expect(initialSupply.sub(newSupply)).to.equal(100);
    });
});
```

## ⛽ Gas Costs

| Function | Estimated Gas |
|----------|--------------|
| Deploy | ~1,500,000 |
| Transfer | ~51,000 |
| Approve | ~44,000 |
| TransferFrom | ~60,000 |
| Mint | ~51,000 |
| Burn | ~35,000 |
| Add/Remove Minter | ~46,000 |

## 🔍 Events

### Standard ERC20 Events
```solidity
event Transfer(address indexed from, address indexed to, uint256 value);
event Approval(address indexed owner, address indexed spender, uint256 value);
```

### Extended Events
```solidity
event MinterAdded(address indexed minter);
event MinterRemoved(address indexed minter);
event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
```

## 🛡 Security Considerations

### Implemented Protections
- Zero address checks
- Balance validation
- Allowance checks
- Access control modifiers
- Overflow protection (Solidity 0.8+)

### Best Practices
- Events for all state changes
- Explicit error messages
- No reentrancy vulnerabilities
- Clear access control

## 🔗 Integration

### Web3.js Example
```javascript
const contract = new web3.eth.Contract(ABI, contractAddress);

// Transfer tokens
await contract.methods.transfer(recipient, amount).send({from: sender});

// Check balance
const balance = await contract.methods.balanceOf(address).call();
```

### Ethers.js Example
```javascript
const contract = new ethers.Contract(address, ABI, signer);

// Transfer tokens
await contract.transfer(recipient, amount);

// Check balance
const balance = await contract.balanceOf(address);
```

## ⚠️ Important Notes

- Not audited - use at own risk
- Test thoroughly before mainnet deployment
- Consider adding pausable functionality for production
- Implement multi-sig for owner role in production

## 📄 License

MIT License