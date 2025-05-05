import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import torch
import torch.nn as nn
from tqdm import tqdm
import uuid
from utils import intermediate_loss
# 蒸馏训练函数
def distill_train(teacher_model, student_model, train_dataloader, val_dataloader, epochs=3, lr=1e-5,device="cuda"):
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
    alpha, beta, gamma = 0.4, 0.3, 0.3
    for epoch in range(epochs):
        student_model.train()
        train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+34}/{epochs+33} - Training"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            mse_last_hidden = nn.MSELoss()(
                student_outputs.last_hidden_state,
                teacher_outputs.last_hidden_state
            )
            mse_pooler = nn.MSELoss()(
                student_outputs.pooler_output,
                teacher_outputs.pooler_output
            )
            mse_intermediate = intermediate_loss(
                student_outputs.hidden_states[6],
                teacher_outputs.hidden_states[12]
            )
            loss = alpha * mse_last_hidden + beta * mse_pooler + gamma * mse_intermediate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_dataloader)

        student_model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+34}/{epochs+33} - Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                teacher_outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                student_outputs = student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                mse_last_hidden = nn.MSELoss()(
                    student_outputs.last_hidden_state,
                    teacher_outputs.last_hidden_state
                )
                mse_pooler = nn.MSELoss()(
                    student_outputs.pooler_output,
                    teacher_outputs.pooler_output
                )
                mse_intermediate = intermediate_loss(
                    student_outputs.hidden_states[6],
                    teacher_outputs.hidden_states[12]
                )
                loss = alpha * mse_last_hidden + beta * mse_pooler + gamma * mse_intermediate
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print(f"Average Validation Loss: {avg_val_loss:.4f}")
        student_model.save_pretrained(f"student_model_epoch{epoch+34}_{str(uuid.uuid4())}")
