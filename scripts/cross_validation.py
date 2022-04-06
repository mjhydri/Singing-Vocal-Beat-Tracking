from sklearn.model_selection import KFold


kf = KFold(n_splits=8)
def train(network, epochs, save_Model = False):
    total_acc = 0
    for fold, (train_index, test_index) in enumerate(kf.split(x_train, y_train)):
        ### Dividing data into folds
        x_train_fold = x_train[train_index]
        x_test_fold = x_train[test_index]
        y_train_fold = y_train[train_index]
        y_test_fold = y_train[test_index]

        train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
        test = torch.utils.data.TensorDataset(x_test_fold, y_test_fold)
        train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
        test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

        for epoch in range(epochs):
            print('\nEpoch {} / {} \nFold number {} / {}'.format(epoch + 1, epochs, fold + 1 , kf.get_n_splits()))
            correct = 0
            network.train()
            for batch_index, (x_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                out = network(x_batch)
                loss = loss_f(out, y_batch)
                loss.backward()
                optimizer.step()
                pred = torch.max(out.data, dim=1)[1]
                correct += (pred == y_batch).sum()
                if (batch_index + 1) % 32 == 0:
                    print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                        (batch_index + 1)*len(x_batch), len(train_loader.dataset),
                        100.*batch_index / len(train_loader), loss.data, float(correct*100) / float(batch_size*(batch_index+1))))
        total_acc += float(correct*100) / float(batch_size*(batch_index+1))
    total_acc = (total_acc / kf.get_n_splits())
    print('\n\nTotal accuracy cross validation: {:.3f}%'.format(total_acc))