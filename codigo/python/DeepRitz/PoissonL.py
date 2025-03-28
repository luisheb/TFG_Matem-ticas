import numpy as np
import math, torch, generateData, time
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR
import torch.nn as nn
import matplotlib.pyplot as plt
import sys, os
import writeSolution


# Network structure
class RitzNet(torch.nn.Module):
    def __init__(self, params):
        super(RitzNet, self).__init__()
        self.params = params
        self.linearIn = nn.Linear(self.params["d"], self.params["width"])
        self.linear = nn.ModuleList()
        for _ in range(params["depth"]):
            self.linear.append(nn.Linear(self.params["width"], self.params["width"]))

        self.linearOut = nn.Linear(self.params["width"], self.params["dd"])

    def forward(self, x):
        x = torch.tanh(self.linearIn(x))  # Match dimension
        for layer in self.linear:
            x_temp = torch.tanh(layer(x))
            x = x_temp

        return self.linearOut(x)


def preTrain(model, device, params, preOptimizer, preScheduler, fun):
    model.train()
    file = open("lossData.txt", "w")

    for step in range(params["preStep"]):
        # The volume integral
        data = (
            torch.from_numpy(
                generateData.sampleFromDisk(params["radius"], params["bodyBatch"])
            )
            .float()
            .to(device)
        )

        output = model(data)

        target = fun(params["radius"], data)

        loss = output - target
        loss = torch.mean(loss * loss) * math.pi * params["radius"] ** 2

        if step % params["writeStep"] == params["writeStep"] - 1:
            with torch.no_grad():
                ref = exact(data)
                error = errorFun(output, ref, params)
                # print("Loss at Step %s is %s."%(step+1,loss.item()))
                print("Error at Step %s is %s." % (step + 1, error))
            file.write(str(step + 1) + " " + str(error) + "\n")

        model.zero_grad()
        loss.backward()

        # Update the weights.
        preOptimizer.step()
        # preScheduler.step()


def train(model, device, params, optimizer, scheduler):
    ratio = (4 * 2.0 + 2 * math.pi * 0.3) / (2.0 * 2.0 - math.pi * 0.3**2)
    model.train()

    data1 = (
        torch.from_numpy(generateData.sampleFromLShape(1, params["bodyBatch"]))
        .float()
        .to(device)
    )
    data2 = (
        torch.from_numpy(generateData.sampleFromLShapeBoundary(1, params["bdryBatch"]))
        .float()
        .to(device)
    )
    x_shift = torch.from_numpy(np.array([params["diff"], 0.0])).float().to(device)
    y_shift = torch.from_numpy(np.array([0.0, params["diff"]])).float().to(device)
    data1_x_shift = data1 + x_shift
    data1_y_shift = data1 + y_shift

    loss_history = []  # Store loss values

    for step in range(params["trainStep"] - params["preStep"]):
        output1 = model(data1)
        output1_x_shift = model(data1_x_shift)
        output1_y_shift = model(data1_y_shift)

        dfdx = (output1_x_shift - output1) / params["diff"]
        dfdy = (output1_y_shift - output1) / params["diff"]

        model.zero_grad()

        # Loss function 1
        fTerm = ffun(data1).to(device)
        loss1 = torch.mean(0.5 * (dfdx * dfdx + dfdy * dfdy) - fTerm * output1)

        # Loss function 2
        output2 = model(data2)
        target2 = exact(data2)
        loss2 = torch.mean(
            (output2 - target2) * (output2 - target2) * params["penalty"] * ratio
        )
        loss = loss1 + loss2
        loss_history.append(loss.item())  # Save loss

        if step % params["writeStep"] == params["writeStep"] - 1:
            with torch.no_grad():
                target = exact(data1)
                error = errorFun(output1, target, params)
                # print("Loss at Step %s is %s."%(step+params["preStep"]+1,loss.item()))
                print(
                    "Loss at Step %s is %s."
                    % (step + params["preStep"] + 1, loss.item())
                )
            file = open("lossData.txt", "a")
            file.write(str(step + params["preStep"] + 1) + " " + str(error) + "\n")

        if step % params["sampleStep"] == params["sampleStep"] - 1:
            data1 = (
                torch.from_numpy(generateData.sampleFromLShape(1, params["bodyBatch"]))
                .float()
                .to(device)
            )
            data2 = (
                torch.from_numpy(
                    generateData.sampleFromLShapeBoundary(1, params["bdryBatch"])
                )
                .float()
                .to(device)
            )

            data1_x_shift = data1 + x_shift
            data1_y_shift = data1 + y_shift

        if 10 * (step + 1) % params["trainStep"] == 0:
            print("%s%% finished..." % (100 * (step + 1) // params["trainStep"]))

        loss.backward()

        optimizer.step()
        scheduler.step()

    return loss_history  # Return loss values


def errorFun(output, target, params):
    error = output - target
    error = math.sqrt(torch.mean(error * error))
    ref = math.sqrt(torch.mean(target * target))

    if ref == 0:  # Avoid division by zero
        return error  # Return absolute error instead of relative error

    return error / ref


def test(model, device, params):
    numQuad = params["numQuad"]

    data = (
        torch.from_numpy(generateData.sampleFromLShape(1, numQuad)).float().to(device)
    )
    output = model(data)
    target = exact(data).to(device)

    error = output - target
    error = math.sqrt(torch.mean(error * error))
    ref = math.sqrt(torch.mean(target * target))

    if ref == 0:  # Avoid division by zero
        return error  # Return absolute error instead of relative error

    return error / ref


def ffun(data):
    # f = 1
    return torch.ones([data.shape[0], 1], dtype=torch.float)


def exact(data):
    return torch.zeros((data.shape[0], 1), dtype=torch.float, device=data.device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# def rough(r,data):
#     output = r**2-r*torch.sum(data*data,dim=1)**0.5
#     return output.unsqueeze(1)


def plot_loss(loss_history):
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(loss_history)), loss_history, linestyle="-")
    plt.xlabel("Iterations (every {} steps)".format(500))
    plt.ylabel("Loss")
    plt.yscale("log")  # Set y-axis to log scale
    plt.title("Training Loss Over Time (Log Scale)")
    plt.grid(
        True, which="both", linestyle="--", linewidth=0.5
    )  # Improve grid visibility
    plt.show()


def main():
    # Parameters
    # torch.manual_seed(21)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = dict()
    params["d"] = 2  # 2D
    params["dd"] = 1  # Scalar field
    params["bodyBatch"] = 1024  # Batch size
    params["bdryBatch"] = 1024  # Batch size for the boundary integral
    params["lr"] = 0.01  # Learning rate
    params["preLr"] = 0.01  # Learning rate (Pre-training)
    params["width"] = 8  # Width of layers
    params["depth"] = 2  # Depth of the network: depth+2
    params["numQuad"] = 40000  # Number of quadrature points for testing
    params["trainStep"] = 10000
    params["penalty"] = 500
    params["preStep"] = 0
    params["diff"] = 0.001
    params["writeStep"] = 500
    params["sampleStep"] = 10
    params["step_size"] = 5000
    params["gamma"] = 0.3
    params["decay"] = 0.00001

    startTime = time.time()
    model = RitzNet(params).to(device)
    print("Generating network costs %s seconds." % (time.time() - startTime))

    preOptimizer = torch.optim.Adam(model.parameters(), lr=params["preLr"])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["decay"]
    )
    scheduler = StepLR(optimizer, step_size=params["step_size"], gamma=params["gamma"])

    startTime = time.time()
    loss_history = train(model, device, params, optimizer, scheduler)
    print("Training costs %s seconds." % (time.time() - startTime))

    model.eval()
    testError = test(model, device, params)
    print("The test error (of the last model) is %s." % testError)
    print("The number of parameters is %s," % count_parameters(model))

    torch.save(model.state_dict(), "last_model.pt")

    # Plot loss graph
    plot_loss(loss_history)

    pltLShapeResult(model, device, 500)


def pltLShapeResult(model, device, nSample):
    xList = np.linspace(0, 1, nSample)
    yList = np.linspace(0, 1, nSample)

    xx, yy = np.meshgrid(xList, yList)
    zz = np.full_like(xx, np.nan)  # Initialize with NaN to mask out unused areas

    for i in range(nSample):
        for j in range(nSample):
            if xx[i, j] < 0.5 or yy[i, j] < 0.5:  # Exclude upper-right quadrant
                coord = np.array([xx[i, j], yy[i, j]])
                zz[i, j] = model(
                    torch.tensor(coord, dtype=torch.float).to(device)
                ).item()

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, zz, levels=100, cmap="viridis", alpha=0.8)
    plt.colorbar(label="Solution Value")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("L-Shape Solution Plot")

    # Overlaying a white patch to mask the removed quadrant
    plt.fill_between([0.5, 1], 0.5, 1, color="white", alpha=1)

    plt.show()


if __name__ == "__main__":
    main()
