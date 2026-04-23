import pytorch_lightning as pl
from transformers import VisionEncoderDecoderModel, AdamW, get_linear_schedule_with_warmup

class DonutFullModel(pl.LightningModule):
    def __init__(self, processor, lr=2e-6):
        """
        PyTorch Lightning Module for the Donut model.
        Args:
            processor: DonutProcessor (contains image processor and tokenizer).
            lr: Base learning rate for the optimizer.
        """
        super().__init__()
        # Save hyperparameters except the complex processor object
        self.save_hyperparameters(ignore=['processor'])
        self.processor = processor
        self.lr = lr
        
        # Initialize the VisionEncoderDecoderModel from pre-trained weights
        self.model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
        
        # Resize decoder embeddings to match the new vocabulary size (including special task tokens)
        self.model.decoder.resize_token_embeddings(len(self.processor.tokenizer))
        
        # Configure model parameters for the specific SROIE extraction task
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.decoder_start_token_id = self.processor.tokenizer.convert_tokens_to_ids('<s_sroie>')
        
        # Enable gradient checkpointing to save VRAM at the cost of slight speed reduction
        self.model.gradient_checkpointing_enable()

    def forward(self, pixel_values, labels):
        """Standard forward pass for VisionEncoderDecoderModel"""
        return self.model(pixel_values, labels=labels)
    
    def training_step(self, batch, batch_idx):
        """Single training iteration: calculate loss and log it"""
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        
        outputs = self(pixel_values, labels)
        loss = outputs.loss
        
        # Log training loss to progress bar and logger (on both step and epoch levels)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation iteration to monitor the model's performance on unseen data"""
        outputs = self(batch["pixel_values"], batch["labels"]) 
        val_loss = outputs.loss
        
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        return val_loss
   
    def configure_optimizers(self):
        """Setup AdamW optimizer and a Linear Scheduler with Warmup"""
        optimizer = AdamW(
            self.parameters(), 
            lr=self.lr,
            weight_decay=0.05  # Applied to prevent overfitting
        )
        
        # Calculate total training steps for the scheduler
        try:
            total_steps = self.trainer.estimated_stepping_batches
        except Exception:
            # Fallback value if trainer is not fully initialized
            total_steps = 10000 
            
        # 10% of total steps used for linear warmup to stabilize training start
        warmup_steps = int(total_steps * 0.1)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # Update learning rate after every batch
            },
        }