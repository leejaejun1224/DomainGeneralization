import torch
import torch.nn as nn
import StereoDepthUDA


def train_step(model, data_batch, optimizer):
    model.train()
    optimizer.zero_grad()
    
    outputs = model.forward_train(data_batch) 
    loss = outputs['loss']
    loss.backward()
    optimizer.step()
    
    # ema update here?
    model.update_ema()
    
    return outputs['log_vars'] 



def main(args):
    
    
    for epoch in range(args.num_epochs):
        for data_batch in train_loader:
            log_vars = train_step(args.model, data_batch, optimizer)
        train_step()
    
    
    
    
    return






if __name__=="__main__":
    # argparser
    main()