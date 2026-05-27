import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


        for epoch in range(epochs):

            torch.manual_seed(epoch)

            X, Y = [], []

            randomIndex = torch.randint(len(data) - context_length, (batch_size, )).tolist()


            for i in randomIndex:

                X.append(data[i:i + context_length])
                Y.append(data[i + 1: i + 1 + context_length])

            X = torch.stack(X).long()
            Y = torch.stack(Y).long()

            
            logits = model(X) # size is (batch_size, context_length, vocab_size)

            

            B, T, C = logits.shape

            logits = logits.reshape(B * T, C) 
            Y = Y.reshape(B * T)               

            loss = F.cross_entropy(logits, Y)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        return round(loss.item(), 4)
            

    
        pass
