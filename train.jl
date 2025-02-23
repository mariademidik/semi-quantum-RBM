include("sqRBM.jl")
using .qbm: train

using Printf

dataset = "bas" #  "bas", "parity", "cardinality", "on2"
model = "rbm" # "rbm", "sqrbm-XZ", "srtbm-XYZ"
num_visible = 4
num_hidden = 2
runID = 1
num_iter = 2000

logPath = "./logs/training/"

train(model, num_visible, num_hidden, runID, logPath, dataset, num_iter)
