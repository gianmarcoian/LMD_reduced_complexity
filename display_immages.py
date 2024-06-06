import torch
import matplotlib.pyplot as plt
import os

def save_combined_images(batch_file, output_dir, num_images=9):
    batch_data = torch.load(batch_file)
    orig_images = batch_data['orig']
    masked_images = batch_data['masked']
    recon_images = batch_data['recon']

    orig_images = orig_images[:num_images]
    masked_images = masked_images[0][:num_images]  
    recon_images = recon_images[0][:num_images]    

    fig, axs = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

    for i in range(num_images):
        orig_img = orig_images[i].permute(1, 2, 0).numpy()
        masked_img = masked_images[i].permute(1, 2, 0).numpy()
        recon_img = recon_images[i].permute(1, 2, 0).numpy()

        orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
        masked_img = (masked_img - masked_img.min()) / (masked_img.max() - masked_img.min())
        recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min())

        axs[i, 0].imshow(orig_img, cmap='gray')
        axs[i, 0].set_title('Original Image')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(masked_img, cmap='gray')
        axs[i, 1].set_title('Masked Image')
        axs[i, 1].axis('off')

        axs[i, 2].imshow(recon_img, cmap='gray')
        axs[i, 2].set_title('Reconstructed Image')
        axs[i, 2].axis('off')
    
    plt.tight_layout()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, 'combined_images.jpeg')
    plt.savefig(output_file, format='jpeg')
    plt.close(fig)

batch_file_path = 'path_to_your_batch.pth'
output_directory = 'path_to_output_directory'

save_combined_images(batch_file_path, output_directory, num_images=9)
