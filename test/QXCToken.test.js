const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("QXCToken", function () {
    let QXCToken;
    let qxcToken;
    let owner;
    let addr1;
    let addr2;
    
    beforeEach(async function () {
        QXCToken = await ethers.getContractFactory("QXCToken");
        [owner, addr1, addr2] = await ethers.getSigners();
        qxcToken = await QXCToken.deploy();
    });
    
    describe("Deployment", function () {
        it("Should set the right owner", async function () {
            expect(await qxcToken.owner()).to.equal(owner.address);
        });
        
        it("Should assign the total supply of tokens to the owner", async function () {
            const ownerBalance = await qxcToken.balanceOf(owner.address);
            expect(await qxcToken.totalSupply()).to.equal(ownerBalance);
        });
        
        it("Should have correct token details", async function () {
            expect(await qxcToken.name()).to.equal("QENEX Token");
            expect(await qxcToken.symbol()).to.equal("QXC");
            expect(await qxcToken.decimals()).to.equal(8);
        });
    });
    
    describe("Staking", function () {
        it("Should allow users to stake tokens", async function () {
            const stakeAmount = ethers.utils.parseUnits("1000", 8);
            
            await qxcToken.transfer(addr1.address, stakeAmount);
            await qxcToken.connect(addr1).stake(stakeAmount);
            
            const stakingInfo = await qxcToken.getStakingInfo(addr1.address);
            expect(stakingInfo.stakedAmount).to.equal(stakeAmount);
        });
        
        it("Should calculate rewards correctly", async function () {
            const stakeAmount = ethers.utils.parseUnits("1000", 8);
            
            await qxcToken.transfer(addr1.address, stakeAmount);
            await qxcToken.connect(addr1).stake(stakeAmount);
            
            // Simulate time passing
            await ethers.provider.send("evm_increaseTime", [365 * 24 * 60 * 60]); // 1 year
            await ethers.provider.send("evm_mine");
            
            const reward = await qxcToken.calculateRewards(addr1.address);
            const expectedReward = stakeAmount.mul(12).div(100); // 12% annual
            
            expect(reward).to.be.closeTo(expectedReward, ethers.utils.parseUnits("1", 8));
        });
    });
});
