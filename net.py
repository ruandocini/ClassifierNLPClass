import torch
from torch import nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np


input_dim = 4 
hidden_layers = 25 
output_dim = 3

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_layers)
        self.linear2 = nn.Linear(hidden_layers, output_dim)
    
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x

    def fit():
        

class Data(Dataset):
  def __init__(self, X_train, y_train):
    # need to convert float64 to float32 else 
    # will get the following error
    # RuntimeError: expected scalar type Double but found Float
    self.X = torch.from_numpy(X_train.astype(np.float32))
    # need to convert float64 to Long else 
    # will get the following error
    # RuntimeError: expected scalar type Long but found Float
    self.y = torch.from_numpy(y_train).type(torch.LongTensor)
    self.len = self.X.shape[0]
  
  def __getitem__(self, index):
    return self.X[index], self.y[index]
  def __len__(self):
    return self.len


if __name__ == "__main__":

    # raw_data = pd.read_excel("data/data.xlsx")

    # word2vec = KeyedVectors.load_word2vec_format("cbow_s100.txt")
    # X = pd.DataFrame(
    #     [avg_document_vector(word2vec, doc) for doc in raw_data["text"]]
    # )
    # X = torch.Tensor(X.values)
    # X = X.type(torch.FloatTensor)

    # Y = categorical_to_numerical(raw_data)["req_type"].values.reshape(1, -1)[0]
    # Y = torch.Tensor(Y)
    # Y = Y.type(torch.FloatTensor)

    X, Y = make_classification(
    n_samples=100, n_features=4, n_redundant=0,
    n_informative=3,  n_clusters_per_class=2, n_classes=3
    )

    X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42)

    traindata = Data(X_train, Y_train)

    batch_size = 4
    trainloader = DataLoader(traindata, batch_size=batch_size, 
                            shuffle=True, num_workers=2)

    clf = Network()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(clf.parameters(), lr=0.1)

    epochs = 10
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            # set optimizer to zero grad to remove previous epoch gradients
            optimizer.zero_grad()
            # forward propagation
            outputs = clf(inputs)
            loss = criterion(outputs, labels)
            # backward propagation
            loss.backward()
            # optimize
            optimizer.step()
            running_loss += loss.item()
        # display statistics
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')

    testdata = Data(X_test, Y_test)
    testloader = DataLoader(testdata, batch_size=batch_size, 
                        shuffle=True, num_workers=2)

    correct, total = 0, 0
    # no need to calculate gradients during inference
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            # calculate output by running through the network
            outputs = clf(inputs)
            # get the predictions
            __, predicted = torch.max(outputs.data, 1)
            # update results
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the {len(testdata)} test data: {100 * correct // total} %')



    # loss = loss_function(pred_y, Y)
    # losses.append(loss.item())
    # plt.plot(losses)
    # plt.ylabel("loss")
    # plt.xlabel("epoch")
    # plt.title("Learning rate %f" % (learning_rate))
    # plt.show()
