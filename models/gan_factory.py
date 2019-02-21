from models import gan, gan_simple, gan_normal, gan_deep

class gan_factory(object):

    @staticmethod
    def generator_factory(type):
        if type == 'simple':
            return gan_simple.generator()
        elif type == 'normal':
            return gan_normal.generator()
        elif type == 'deep':
            return gan_deep.generator()
        elif type == 'vanilla_gan':
            return gan.generator()

    @staticmethod
    def discriminator_factory(type):
        if type == 'simple':
            return gan_simple.discriminator()
        elif type == 'normal':
            return gan_normal.discriminator()
        elif type == 'deep':
            return gan_deep.discriminator()
        elif type == 'vanilla_gan':
            return gan.discriminator()
