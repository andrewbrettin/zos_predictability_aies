configs = {
    'mean': dict(
        network_architecture=[10, 10],
        lr=1e-5,
        optimizer="Adam",
        l2=0.0,
        dropout=0.1,
        epochs=100,
        early_stopping=True,
        patience=10,
        batch_size=32,
    ),
    'residual': dict(
        network_architecture=[10, 10],
        lr=1e-6,
        optimizer="Adam",
        l2=0.0,
        dropout=0.1,
        epochs=100,
        early_stopping=True,
        patience=10,
        batch_size=32,
    ),
}