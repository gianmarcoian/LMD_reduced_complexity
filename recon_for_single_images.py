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

# Data related args-> from batch_size to num_images
flags.DEFINE_integer("num_images_in", 10, "number of in-domain images to reconstruct")
flags.DEFINE_integer("num_images_out", 10, "number of out-of-domain images to reconstruct")
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

    def load_images_from_folder(folder, num_images):
        dataset = ImageFolder(root=folder, transform=transform)
        images = [dataset[i][0] for i in range(min(num_images, len(dataset)))]
        return images

    if FLAGS.in_domain in DATA_TO_PATH:
        pos_images = load_images_from_folder(DATA_TO_PATH[FLAGS.in_domain], FLAGS.num_images_in)
    else:
        raise NotImplementedError(f"In-domain dataset {FLAGS.in_domain} not recognized.")
    
    if FLAGS.out_of_domain in DATA_TO_PATH:
        neg_images = load_images_from_folder(DATA_TO_PATH[FLAGS.out_of_domain], FLAGS.num_images_out)
    else:
        raise NotImplementedError(f"Out-of-domain dataset {FLAGS.out_of_domain} not recognized.")

    return pos_images, neg_images


def get_mask_info_dict():
    return {
        "mask_type": FLAGS.mask_type, 
        "image_size": FLAGS.config.data.image_size,
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

        self.shape = ( self.config.data.num_channels,
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
    def recon(self, image, i, mask_info_dict=None, mode="pos"):
        mask = get_mask(
            mask_type=mask_info_dict["mask_type"],
            image_size=mask_info_dict["image_size"],
            num_channels=mask_info_dict["num_channels"],
            mask_file_path=mask_info_dict["mask_file_path"],
            checkerboard_num_blocks=mask_info_dict["checkerboard_num_blocks"],
            rand_patch_size=mask_info_dict["rand_patch_size"],
            rand_mask_ratio=mask_info_dict["rand_mask_ratio"],
            maskgen=mask_info_dict["maskgen"],
            maskgen_offset=mask_info_dict["maskgen_offset"]
        ).cuda()

        if FLAGS.save_mask:
            save_mask(
                mask=mask, 
                info_dict=mask_info_dict, 
                save_dir="{workdir}/{mode}/mask".format(workdir=FLAGS.workdir, mode=mode), 
                compress=True,
                identifier="{i}_{j}".format(i=i, j=mask_info_dict['maskgen_offset'])
            )

        image_masked = image * mask
        image_inpainted = self.inpainter(self.model, self.scaler(image.cuda()), mask.cuda())
        return image_masked.detach().cpu(), image_inpainted.detach().cpu()

def main(argv):
    pos_images, neg_images = get_datasets()
    print(f"Number of examples in pos_images: {len(pos_images)}")
    print(f"Number of examples in neg_images: {len(neg_images)}")

    mask_info_dict = get_mask_info_dict()
    if FLAGS.mask_type == "random":
        maskgen = MaskGenerator(
            input_size=mask_info_dict["image_size"], 
            mask_patch_size=mask_info_dict["rand_patch_size"],
            model_patch_size=1, 
            mask_ratio=mask_info_dict["rand_mask_ratio"]
        )
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
        for i, image in enumerate(pos_images):
            image = image.unsqueeze(0).cuda()
            save_dict = {"orig": image.detach().cpu(), "masked": [], "recon": []}

            for j in range(FLAGS.reps_per_image):
                mask_info_dict['maskgen_offset'] = j
                masked, recon = detector.recon(batch=image, i=i, mask_info_dict=mask_info_dict, mode="pos")
                save_dict["masked"].append(masked)
                save_dict["recon"].append(recon)

            torch.save(save_dict, "%s/pos/image_%d.pth" % (FLAGS.workdir, i))
            del save_dict

        for i, image in enumerate(neg_images):
            image = image.unsqueeze(0).cuda()
            save_dict = {"orig": image.detach().cpu(), "masked": [], "recon": []}

            for j in range(FLAGS.reps_per_image):
                mask_info_dict['maskgen_offset'] = j
                masked, recon = detector.recon(batch=image, i=i, mask_info_dict=mask_info_dict, mode="neg")
                save_dict["masked"].append(masked)
                save_dict["recon"].append(recon)

            torch.save(save_dict, "%s/neg/image_%d.pth" % (FLAGS.workdir, i))
            del save_dict

if __name__ == "__main__":
    app.run(main)
