const hre = require("hardhat");

async function main() {
    console.log("Deploying QXC Token...");
    
    const QXCToken = await hre.ethers.getContractFactory("QXCToken");
    const qxcToken = await QXCToken.deploy();
    
    await qxcToken.deployed();
    
    console.log("QXC Token deployed to:", qxcToken.address);
    
    // Verify contract
    if (hre.network.name !== "hardhat" && hre.network.name !== "localhost") {
        console.log("Waiting for block confirmations...");
        await qxcToken.deployTransaction.wait(6);
        
        console.log("Verifying contract...");
        await hre.run("verify:verify", {
            address: qxcToken.address,
            constructorArguments: [],
        });
    }
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
