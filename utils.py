import torch.nn as nn
def intermediate_loss(student_hidden, teacher_hidden):
    return nn.MSELoss()(student_hidden, teacher_hidden)

