# adversarial attack methods

## FGSM
> acronym for fast gradient sign method
```
def FGSM(eps, images, target, model, criterion):
    X = images.clone()
    X.requires_grad = True
    output = model(X)
    loss = criterion(output, target)
    loss.backward()
    grad_sign = X.grad.data.sign()
    return (X + eps*grad_sign).clamp(0, 1)
```


## PGD
> acronym for projected gradient descent
```
def PGD(eps, images, target, model, criterion):
    X_orig = images.clone()    
    X_var = images.clone()
    for __ in range(40):
        X = X_var.clone()
        X.requires_grad = True
        output = model(X)
        loss = criterion(output, target)
        loss.backward()
        grad_sign = X.grad.data.sign()
        X_var = X_var + 0.05*grad_sign
        # X_var.clamp(X_orig-eps, X_orig+eps)
        X_var = torch.where(X_var < X_orig-eps, X_orig-eps, X_var)
        X_var = torch.where(X_var > X_orig+eps, X_orig+eps, X_var)
        X_var.clamp(0, 1)
    return X_var
```

## todo
> more explanations