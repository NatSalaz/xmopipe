from models.t2m_eval_modules import *
from utils.word_vectorizer import POS_enumerator
from os.path import join as pjoin


def build_models(config, evaluator="text_mot_match"):
    movement_enc = MovementConvEncoder(
        config.dim_pose - 4, config.dim_movement_enc_hidden, config.dim_movement_latent
    )
    text_enc = TextEncoderBiGRUCo(
        word_size=config.dim_word,
        pos_size=config.dim_pos_ohot,
        hidden_size=config.dim_text_hidden,
        output_size=config.dim_coemb_hidden,
        device=config.device,
    )

    motion_enc = MotionEncoderBiGRUCo(
        input_size=config.dim_movement_latent,
        hidden_size=config.dim_motion_hidden,
        output_size=config.dim_coemb_hidden,
        device=config.device,
    )

    checkpoint = torch.load(
        pjoin(
            config.checkpoints_dir,
            config.dataset_name,
            evaluator,
            "model",
            "finest.tar",
        ),
        map_location=config.device,
    )
    movement_enc.load_state_dict(checkpoint["movement_encoder"])
    text_enc.load_state_dict(checkpoint["text_encoder"])
    motion_enc.load_state_dict(checkpoint["motion_encoder"])
    print(
        "Loading Evaluation Model from %s "
        % (
            pjoin(
                config.checkpoints_dir,
                config.dataset_name,
                evaluator,
                "model",
                "finest.tar",
            )
        )
    )
    return text_enc, motion_enc, movement_enc


class EvaluatorModelWrapper(object):

    def __init__(self, config):


        if config.dataset_name == 't2m' or config.dataset_name == 'idea400':
            config.dim_pose = 263
        elif config.dataset_name == "kit":
            config.dim_pose = 251
        else:
            raise KeyError("Dataset not Recognized!!!")

        config.dim_word = 300
        config.max_motion_length = 196
        config.dim_pos_ohot = len(POS_enumerator)
        config.dim_motion_hidden = 1024
        config.max_text_len = 20
        config.dim_text_hidden = 512
        config.dim_coemb_hidden = 512

        # print(config)

        self.text_encoder, self.motion_encoder, self.movement_encoder = build_models(
            config,
            evaluator=(
                "text_mot_match"
                if not hasattr(config, "evaluator")
                else config.evaluator
            ),
        )
        self.config = config
        self.device = config.device

        self.text_encoder.to(config.device)
        self.motion_encoder.to(config.device)
        self.movement_encoder.to(config.device)

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()

    # Please note that the results does not follow the order of inputs
    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            """Movement Encoding"""
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.config.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)

            """Text Encoding"""
            text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            text_embedding = text_embedding[align_idx]
        return text_embedding, motion_embedding

    # Please note that the results does not follow the order of inputs
    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            """Movement Encoding"""
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.config.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding


## Borrowed form MDM
# our version
def build_evaluators(config):
    movement_enc = MovementConvEncoder(
        config["dim_pose"] - 4,
        config["dim_movement_enc_hidden"],
        config["dim_movement_latent"],
    )
    text_enc = TextEncoderBiGRUCo(
        word_size=config["dim_word"],
        pos_size=config["dim_pos_ohot"],
        hidden_size=config["dim_text_hidden"],
        output_size=config["dim_coemb_hidden"],
        device=config["device"],
    )

    motion_enc = MotionEncoderBiGRUCo(
        input_size=config["dim_movement_latent"],
        hidden_size=config["dim_motion_hidden"],
        output_size=config["dim_coemb_hidden"],
        device=config["device"],
    )

    ckpt_dir = config["dataset_name"]
    if config["dataset_name"] == "humanml":
        ckpt_dir = "t2m"

    checkpoint = torch.load(
        pjoin(
            config["checkpoints_dir"], ckpt_dir, "text_mot_match", "model", "finest.tar"
        ),
        map_location=config["device"],
    )
    movement_enc.load_state_dict(checkpoint["movement_encoder"])
    text_enc.load_state_dict(checkpoint["text_encoder"])
    motion_enc.load_state_dict(checkpoint["motion_encoder"])
    print(
        "Loading Evaluation Model Wrapper (Epoch %d) Completed!!"
        % (checkpoint["epoch"])
    )
    return text_enc, motion_enc, movement_enc


# our wrapper
class EvaluatorWrapper(object):

    def __init__(self, dataset_name, device):
        config = {
            "dataset_name": dataset_name,
            "device": device,
            "dim_word": 300,
            "max_motion_length": 196,
            "dim_pos_ohot": len(POS_enumerator),
            "dim_motion_hidden": 1024,
            "max_text_len": 20,
            "dim_text_hidden": 512,
            "dim_coemb_hidden": 512,
            "dim_pose": 263 if dataset_name == "humanml" else 251,
            "dim_movement_enc_hidden": 512,
            "dim_movement_latent": 512,
            "checkpoints_dir": "./checkpoints",
            "unit_length": 4,
        }

        self.text_encoder, self.motion_encoder, self.movement_encoder = (
            build_evaluators(config)
        )
        self.config = config
        self.device = config["device"]

        self.text_encoder.to(config["device"])
        self.motion_encoder.to(config["device"])
        self.movement_encoder.to(config["device"])

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()

    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            """Movement Encoding"""
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.config["unit_length"]
            motion_embedding = self.motion_encoder(movements, m_lens)
            # print(motions.shape, movements.shape, motion_embedding.shape, m_lens)

            """Text Encoding"""
            text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            text_embedding = text_embedding[align_idx]
        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            """Movement Encoding"""
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.config["unit_length"]
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding
