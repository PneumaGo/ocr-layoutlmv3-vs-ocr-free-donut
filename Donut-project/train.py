import pytorch_lightning as pl
from transformers import DonutProcessor

if __name__ == "__main__":
    # Initialize the base Donut processor from Hugging Face
    model_name = "naver-clova-ix/donut-base"
    processor = DonutProcessor.from_pretrained(model_name)

    # Set high-resolution input size for better document detail extraction
    # We disable long axis alignment to keep aspect ratios consistent
    processor.image_processor.size = {"height": 1280, "width": 960}
    processor.image_processor.do_align_long_axis = False 

    # Define custom task-specific tokens for the SROIE dataset fields
    new_tokens = [
        "<s_company>", "</s_company>", 
        "<s_date>", "</s_date>", 
        "<s_address>", "</s_address>", 
        "<s_total>", "</s_total>",
        "<s_sroie>", "</s_sroie>" 
    ]
    processor.tokenizer.add_tokens(new_tokens)

    # Initialize the Lightning Module with a specified learning rate
    model = DonutFullModel(processor=processor, lr=2e-5)

    # Data paths (Note: These should be updated relative to the repo root for local runs)
    base_path = "/kaggle/input/datasets/maxbegal/dataset/data"
    
    train_img = f"{base_path}/train/img"
    train_ent = f"{base_path}/train/entities"
    test_img = f"{base_path}/val/img"
    test_ent = f"{base_path}/val/entities"

    # Initialize DataModule for handling train/val splits and loaders
    dm = SROIEDataModule(
        train_img_dir=train_img,
        train_ent_dir=train_ent,
        test_img_dir=test_img,
        test_ent_dir=test_ent,
        processor=processor,
        batch_size=1  # Low batch size due to high-resolution images and VRAM limits
    )

    # Define training callbacks for experiment management and regularization
    callbacks = [
        # Stop training if validation loss stops improving to prevent overfitting
        EarlyStopping(
            monitor="val_loss", 
            patience=8, 
            mode="min"
        ),
        # Save only the best performing model checkpoint
        ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints",
            filename="best-donut-{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
            mode="min",
        ),
        # Log learning rate changes during the warmup/linear decay phases
        LearningRateMonitor(logging_interval="step"),
        
        # SWA for better generalization and more stable convergence
        StochasticWeightAveraging(swa_lrs=2e-6) 
    ]
    
    # Configure the PyTorch Lightning Trainer with optimization features
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",         # Use Mixed Precision (FP16) to speed up training and save memory
        accumulate_grad_batches=8,    # Gradient Accumulation: effectively mimics a larger batch size (1*8=8)
        max_epochs=20,
        gradient_clip_val=1.0,        # Clip gradients to prevent exploding gradient issues
        callbacks=callbacks,  
        log_every_n_steps=10
    )
   
    # Reinforce memory efficiency during training
    model.model.gradient_checkpointing_enable()

    print("Starting FULL Fine-tuning on SROIE dataset...")
    trainer.fit(model, datamodule=dm)

    # Save final artifacts (model weights and processor) for inference or deployment
    output_dir = "donut_sroie_final_model"
    model.model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    print(f"Training complete. Model and processor saved to '{output_dir}'.")