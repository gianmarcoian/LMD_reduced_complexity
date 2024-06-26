from recon_utils import *
from mask_utils import *
import datasets
from models import ddpm, ncsnpp
from sampling import get_predictor, get_corrector
from controllable_generation import get_pc_inpainter

from absl import flags
from ml_collections.config_flags import config_flags
from absl import app

from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)

flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("ckpt_path", None, "checkpoint path to load")
flags.DEFINE_integer("reps_per_image", 1, 'number of recons to do for each image')

# Data related args
flags.DEFINE_integer("batch_size", 100, "batch size to use")

flags.DEFINE_string("in_domain", None, 'in-domain dataset')
flags.DEFINE_string("out_of_domain", None, 'out of domain dataset')
flags.DEFINE_boolean('id_center_crop', True, 'apply center crop to in domain images')
flags.DEFINE_boolean('ood_center_crop', True, 'apply center crop to out of domain images')

# Mask related args
flags.DEFINE_string("mask_type", 'center', 'mask type; center, random, checkerboard, checkerboard_alt')
flags.DEFINE_boolean("save_mask", False, "save mask or not")
flags.DEFINE_integer("mask_num_blocks", 4, 'number of blocks per edge for checkerboard mask; image_size must be divisible by it')
flags.DEFINE_integer("mask_patch_size", 4, 'patch size, if mask type is random')
flags.DEFINE_float("mask_ratio", 0.5, 'mask ratio, if mask type is random')
flags.DEFINE_string("mask_file_path", None, "file path to load mask from")

flags.DEFINE_string("mask_save_dir", None, "directory to save mask")
flags.DEFINE_integer("mask_identifier", 0, "mask identifier")

flags.mark_flags_as_required(["workdir", "config", "ckpt_path", "in_domain", "out_of_domain"])

def get_datasets():
    DATA_TO_PATH = {
        'MNIST_IN': "./data/mnist_in",
        'MNIST_OUT': "./data/mnist_out",
    }
    
    image_size = FLAGS.config.data.image_size
    transform = T.Compose([
        T.Grayscale(num_output_channels=1),  # Assicura che le immagini siano in scala di grigi
        T.Resize(image_size),
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1))  # Ripeti i canali per ottenere un'immagine RGB
    ])

    if FLAGS.in_domain in DATA_TO_PATH:
        pos_dataset = ImageFolder(
            root=DATA_TO_PATH[FLAGS.in_domain],
            transform=transform
        )
    else:
        raise NotImplementedError(f"In-domain dataset {FLAGS.in_domain} not recognized.")
    
    if FLAGS.out_of_domain in DATA_TO_PATH:
        neg_dataset = ImageFolder(
            root=DATA_TO_PATH[FLAGS.out_of_domain],
            transform=transform
        )
    else:
        raise NotImplementedError(f"Out-of-domain dataset {FLAGS.out_of_domain} not recognized.")

    return pos_dataset, neg_dataset

def get_mask_info_dict():
    return {
        "mask_type": FLAGS.mask_type, 
        "image_size": FLAGS.config.data.image_size,
        "batch_size": FLAGS.batch_size,
        "num_channels": 3,
        "mask_file_path": FLAGS.mask_file_path,
        "checkerboard_num_blocks": FLAGS.mask_num_blocks,
        "rand_patch_size": FLAGS.mask_patch_size,
        "rand_mask_ratio": FLAGS.mask_ratio,
        "maskgen": None,
        "maskgen_offset": 0
    }

class Detector(object):
    def __init__(self):
        self.config = FLAGS.config
    
        self.sde, self.eps = get_sde_and_eps(self.config)
        print("loading from checkpoint: {ckpt_path}".format(ckpt_path=FLAGS.ckpt_path))
        assert os.path.exists(FLAGS.ckpt_path)
        self.model = get_model_ema(FLAGS.config, FLAGS.ckpt_path)
        self.model.eval()

        assert FLAGS.mask_type in ['center', 'checkerboard', 'random', 'checkerboard_alt']

        self.shape = (FLAGS.batch_size, self.config.data.num_channels,
                      self.config.data.image_size, self.config.data.image_size)

        self.scaler = datasets.get_data_scaler(self.config)
        self.inverse_scaler = datasets.get_data_inverse_scaler(self.config)

        self.predictor = get_predictor(self.config.sampling.predictor.lower())
        self.corrector = get_corrector(self.config.sampling.corrector.lower())

        self.inpainter = get_pc_inpainter(
            sde=self.sde, 
            predictor=self.predictor, 
            corrector=self.corrector, 
            inverse_scaler=self.inverse_scaler, 
            snr=self.config.sampling.snr,
            n_steps=self.config.sampling.n_steps_each, 
            probability_flow=self.config.sampling.probability_flow, 
            continuous=self.config.training.continuous,
            denoise=self.config.sampling.noise_removal, 
            eps=self.eps)
    
    def recon(self, batch, i, mask_info_dict=None, mode="pos"):
        mask_info_dict['batch_size'] = batch.shape[0]
        mask = get_mask(**mask_info_dict).cuda()
        if FLAGS.save_mask:
            save_mask(mask=mask, 
                      info_dict=mask_info_dict, 
                      save_dir="{workdir}/{mode}/mask".format(workdir=FLAGS.workdir, mode=mode), 
                      compress=True,
                      identifier="{i}_{j}".format(i=i, j=mask_info_dict['maskgen_offset']))
        
        batch_masked = batch * mask
        batch_inpainted = self.inpainter(self.model, self.scaler(batch.cuda()), mask.cuda())
        return batch_masked.detach().cpu(), batch_inpainted.detach().cpu()

def main(argv):
    pos_dataset, neg_dataset = get_datasets()
    print(f"Number of examples in pos_dataset: {len(pos_dataset)}")
    print(f"Number of examples in neg_dataset: {len(neg_dataset)}")

    pos_loader = DataLoader(pos_dataset,
                            drop_last=False, 
                            batch_size=FLAGS.batch_size, 
                            shuffle=False, 
                            num_workers=torch.cuda.device_count())
    neg_loader = DataLoader(neg_dataset, 
                            drop_last=False, 
                            batch_size=FLAGS.batch_size, 
                            shuffle=False, 
                            num_workers=torch.cuda.device_count())
    print("Dataloaders created successfully.")

    mask_info_dict = get_mask_info_dict()
    if FLAGS.mask_type == "random":
        maskgen = MaskGenerator(input_size=mask_info_dict["image_size"], 
                                mask_patch_size=mask_info_dict["rand_patch_size"],
                                model_patch_size=1, 
                                mask_ratio=mask_info_dict["rand_mask_ratio"])
        mask_info_dict['maskgen'] = maskgen

    FLAGS.workdir = "{workdir}/{mask_name}_reps{n_reps}".format(
        workdir=FLAGS.workdir,
        mask_name=get_mask_name(mask_info_dict),
        n_reps=FLAGS.reps_per_image
    )

    os.makedirs("%s/pos" % FLAGS.workdir, exist_ok=True)
    os.makedirs("%s/neg" % FLAGS.workdir, exist_ok=True)

    detector = Detector()
    print("Detector initialized successfully.")

    with torch.no_grad():
     for i, (batch, _) in enumerate(pos_loader):
        print(f"Processing batch {i} for pos_loader.")
        if os.path.exists("%s/pos/batch_%d.pth" % (FLAGS.workdir, i)):
            print(f"Skipping batch {i} for pos_loader as it already exists.")
            continue

        print(f"Batch {i} loaded successfully for pos_loader.")
        batch = batch.cuda()
        save_dict = {"orig": batch.detach().cpu(), "masked": [], "recon": []}

        for j in range(FLAGS.reps_per_image):
            print(f"Reconstruction {j} for batch {i} in pos_loader.")
            mask_info_dict['maskgen_offset'] = j
            masked, recon = detector.recon(batch=batch, i=i, mask_info_dict=mask_info_dict, mode="pos")
            save_dict["masked"].append(masked)
            save_dict["recon"].append(recon)

        print(f"Saving batch {i} for pos_loader.")
        torch.save(save_dict, "%s/pos/batch_%d.pth" % (FLAGS.workdir, i))
        del save_dict

     for i, (batch, _) in enumerate(neg_loader):
        print(f"Processing batch {i} for neg_loader.")
        if os.path.exists("%s/neg/batch_%d.pth" % (FLAGS.workdir, i)):
            print(f"Skipping batch {i} for neg_loader as it already exists.")
            continue

        print(f"Batch {i} loaded successfully for neg_loader.")
        batch = batch.cuda()
        save_dict = {"orig": batch.detach().cpu(), "masked": [], "recon": []}

        for j in range(FLAGS.reps_per_image):
            print(f"Reconstruction {j} for batch {i} in neg_loader.")
            mask_info_dict['maskgen_offset'] = j
            masked, recon = detector.recon(batch=batch, i=i, mask_info_dict=mask_info_dict, mode="neg")
            save_dict["masked"].append(masked)
            save_dict["recon"].append(recon)

        print(f"Saving batch {i} for neg_loader.")
        torch.save(save_dict, "%s/neg/batch_%d.pth" % (FLAGS.workdir, i))
        del save_dict


if __name__ == "__main__":
    app.run(main)
