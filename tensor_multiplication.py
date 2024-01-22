import torch

def create_tensor_of_val(dimensions, val):
    """
    Create a tensor of the given dimensions, filled with the value of `val`.
    dimentions is a tuple of integers.
    Hint: use torch.ones and multiply by val, or use torch.zeros and add val.
    e.g. if dimensions = (2, 3), and val = 3, then the returned tensor should be of shape (2, 3)
    specifically, it should be:
    tensor([[3., 3., 3.], [3., 3., 3.]])
    """
    # Using torch.ones
    res = torch.ones(dimensions) * val  
    return res

def calculate_elementwise_product(A, B):
    """
    Calculate the elementwise product of the two tensors A and B.
    Note that the dimensions of A and B should be the same.
    """
    
    # Calculate elementwise product using torch.mul
    res = torch.mul(A, B)  # TODO: implement this function 
    return res 


def calculate_matrix_product(X, W):
    """
    Calculate the product of the two tensors X and W. ( sum {x_i * w_i})
    Note that the dimensions of X and W should be compatible for multiplication.
    e.g: if X is a tensor of shape (1,3) then W could be a tensor of shape (N,3) i.e: (1,3) or (2,3) etc. but in order for 
         matmul to work, we need to multiply by W.T (W transpose) so that the `inner` dimensions are the same.
    Hint: use torch.matmul to calculate the product.
          This allows us to use a batch of inputs, and not just a single input.
          Also, it allows us to use the same function for a single neuron or multiple neurons.
         
    """
    
    # Calculate matrix product using torch.matmul
    res = torch.matmul(X, W.T)  # Transpose W to match the inner dimensions  
    return res

def calculate_matrix_prod_with_bias(X, W, b):
    """
    Calculate the product of the two tensors X and W. ( sum {x_i * w_i}) and add the bias.
    Note that the dimensions of X and W should be compatible for multiplication.
    e.g: if X is a tensor of shape (1,3) then W could be a tensor of shape (N,3) i.e: (1,3) or (2,3) etc. but in order for
         matmul to work, we need to multiply by W.T (W transpose) so that the `inner` dimensions are the same.
    Hint: use torch.matmul to calculate the product.
          This allows us to use a batch of inputs, and not just a single input.
          Also, it allows us to use the same function for a single neuron or multiple neurons.
       """ 
    # Check if the dimensions of X and W are compatible for matrix multiplication
    assert X.shape[1] == W.shape[1], "Inner dimensions must match for matrix multiplication"

    # Calculate matrix product using torch.matmul
    product = torch.matmul(X, W.T)  # Transpose W to match the inner dimensions

    # Add bias
    res = product + b
    
    return res

def calculate_activation(sum_total):
    """
    Calculate a step function as an activation of the neuron.
    Hint: use PyTorch `heaviside` function.
    """
     
    # Convert scalar value to tensor
    values = torch.tensor(0.0)

    # Use PyTorch heaviside function to implement step function
    res = torch.heaviside(sum_total, values=values)
    return res

def calculate_output(X, W, b):
    """
    Calculate the output of the neuron.
    Hint: use the functions you implemented above.
    """
    
    # Calculate the matrix product with bias
    sum_total = calculate_matrix_prod_with_bias(X, W, b)

    # Apply activation function
    res = calculate_activation(sum_total)
    return res