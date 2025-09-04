# Token Smart Contract

ERC20-compatible token implementation.

## Contract Overview

```solidity
contract Token {
    // Core ERC20 functions
    function transfer(address to, uint256 amount) returns (bool)
    function approve(address spender, uint256 amount) returns (bool)
    function transferFrom(address from, address to, uint256 amount) returns (bool)
    
    // View functions
    function balanceOf(address account) view returns (uint256)
    function allowance(address owner, address spender) view returns (uint256)
    function totalSupply() view returns (uint256)
}
```

## Deployment

### Using Hardhat
```javascript
const Token = await ethers.getContractFactory("Token");
const token = await Token.deploy(
    "Token Name",     // name
    "TKN",           // symbol
    18,              // decimals
    1000000          // initial supply
);
```

### Using Remix
1. Copy contract code to Remix
2. Compile with Solidity 0.8.20
3. Deploy with constructor parameters

## Features

### Security
- Zero address validation
- Balance overflow protection
- Allowance checks
- Event logging

### Standards
- ERC20 compliant
- OpenZeppelin compatible
- Gas optimized

## Testing

```javascript
describe("Token", function() {
    it("Should transfer tokens", async function() {
        const [owner, addr1] = await ethers.getSigners();
        const token = await Token.deploy("Test", "TST", 18, 1000);
        
        await token.transfer(addr1.address, 100);
        expect(await token.balanceOf(addr1.address)).to.equal(100);
    });
});
```

## Gas Costs

| Operation | Gas Used |
|-----------|----------|
| Deploy | ~800,000 |
| Transfer | ~65,000 |
| Approve | ~45,000 |
| TransferFrom | ~75,000 |

## Integration

### Web3.js
```javascript
const contract = new web3.eth.Contract(ABI, address);
await contract.methods.transfer(recipient, amount).send({from: sender});
```

### Ethers.js
```javascript
const contract = new ethers.Contract(address, ABI, signer);
await contract.transfer(recipient, amount);
```

## License

MIT