import mindspore as ms
from mindspore import Tensor, nn, ops

from .lpips import LPIPS


class DiscriminatorWithLoss(nn.Cell):
    """
    Training logic:
        For training step i, input data x:
            1. AE generator takes input x, feedforward to get posterior/latent and reconstructed data, and compute ae loss
            2. AE optimizer updates AE trainable params
            3. D takes the same input x, feed x to AE again **again** to get
                the new posterior and reconstructions (since AE params has updated), feed x and recons to D, and compute D loss
            4. D optimizer updates D trainable params
            --> Go to next training step
        Ref: sd-vae training
    """

    def __init__(
        self,
        autoencoder,
        discriminator,
        disc_start=50001,
        disc_factor=1.0,
        disc_loss="hinge",
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.discriminator = discriminator
        self.disc_start = disc_start
        self.disc_factor = disc_factor

        assert disc_loss in ["hinge", "vanilla"]
        if disc_loss == "hinge":
            self.disc_loss = self.hinge_loss
        else:
            self.softplus = ops.Softplus()
            self.disc_loss = self.vanilla_d_loss

    def hinge_loss(self, logits_real, logits_fake):
        loss_real = ops.mean(ops.relu(1.0 - logits_real))
        loss_fake = ops.mean(ops.relu(1.0 + logits_fake))
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss

    def vanilla_d_loss(self, logits_real, logits_fake):
        d_loss = 0.5 * (ops.mean(self.softplus(-logits_real)) + ops.mean(self.softplus(logits_fake)))
        return d_loss

    def construct(self, x: ms.Tensor, global_step=-1, cond=None):
        """
        Second pass
        Args:
            x: input image/video, (bs c h w)
            weights: sample weights
        """

        # 1. AE forward
        recons, mean = ops.stop_gradient(self.autoencoder(x))

        # 2. Disc forward to get class prediction on real input and reconstrucions
        if cond is None:
            logits_real = self.discriminator(x)
            logits_fake = self.discriminator(recons)
        else:
            logits_real = self.discriminator(ops.concat((x, cond), dim=1))
            logits_fake = self.discriminator(ops.concat((recons, cond), dim=1))

        # TODO: skip previous computation if global step < self.disc_start, to save time
        if global_step >= self.disc_start:
            disc_factor = self.disc_factor
        else:
            disc_factor = 0.0

        d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

        return d_loss


def l2(x, y):
    return ops.pow((x - y), 2)


class VQGeneratorWithLoss(nn.Cell):
    def __init__(
        self,
        autoencoder,
        disc_start=1,
        codebook_weight=1.0,
        pixelloss_weight=1.0,
        disc_weight=1.0,
        disc_factor=1.0,
        perceptual_weight=1.0,
        n_classes=None,
        pixel_loss="l1",
        discriminator=None,
    ):
        super().__init__()

        # build perceptual models for loss compute
        self.autoencoder = autoencoder
        # TODO: set dtype for LPIPS ?
        self.perceptual_loss = LPIPS()  # freeze params inside
        if pixel_loss == "l1":
            self.pixel_loss = nn.L1Loss(reduction="none")
        else:
            self.pixel_loss = l2

        self.codebook_weight = codebook_weight
        self.pixelloss_weight = pixelloss_weight

        self.disc_start = disc_start
        self.disc_weight = disc_weight
        self.disc_factor = disc_factor
        self.perceptual_weight = perceptual_weight

        self.discriminator = discriminator
        # assert discriminator is None, "Discriminator is not supported yet"
        self.n_classes = n_classes

    def loss_function(self, x, codebook_loss, recons, global_step: ms.Tensor = -1, cond=None):
        if not codebook_loss:
            codebook_loss = Tensor([0.0])
        # 2.1 reconstruction loss in pixels
        rec_loss = self.pixel_loss(x, recons)

        # 2.2 perceptual loss
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(x, recons)
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = Tensor([0.0])

        loss = ops.mean(rec_loss)

        # 2.4 discriminator loss if enabled
        # TODO: how to get global_step?
        g_loss = 0.0
        d_weight = self.disc_weight
        if global_step >= self.disc_start:
            if (self.discriminator is not None) and (self.disc_factor > 0.0):
                # calc gan loss
                if cond is None:
                    logits_fake = self.discriminator(recons)
                else:
                    logits_fake = self.discriminator(ops.concat((recons, cond), dim=1))
                g_loss = -ops.mean(logits_fake)
                # TODO: do adaptive weighting based on grad
                # d_weight = self.calculate_adaptive_weight(mean_nll_loss, g_loss, last_layer=last_layer)
        loss += d_weight * self.disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()
        return loss

    # in graph mode, construct code will run in graph. TODO: in pynative mode, need to add ms.jit decorator
    def construct(self, x: ms.Tensor, global_step: ms.Tensor = -1, cond=None):
        """
        x: input image/video, (bs c h w)
        global_step: global training step
        """

        recons, qloss = self.autoencoder(x)
        # 2. compuate loss
        loss = self.loss_function(x, qloss, recons, global_step, cond)

        return loss
