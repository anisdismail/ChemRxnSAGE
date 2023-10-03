class LSTMLMGenerator:
    def __init__(self, config):
        self.config = config
        # TODO: add model loading

    def generate_samples(self):
        model.eval()
        samples = []
        for _ in range(int(self.config.n_samples / self.config.batch_size)):
            sample = model.sample(
                self.config.batch_size, self.config.g_seq_len).cpu().data.numpy().tolist()
            samples.extend(sample)
        with open(output_file, 'w', encoding="utf-8") as fout:
            lines_to_write = [
                ' '.join(map(str, sample)) + '\n' for sample in samples]
            fout.writelines(lines_to_write)


class VAEGenerator:
    def __init__(self, config):
        self.config = config
        # TODO: add model loading
        # TODO: add data to reconstruct from

    def reconstruct(self):
        with open(fname, "w", encoding="utf-8") as fout:
            for batch_data, _ in self.data_iter:
                decoded_batch = model.reconstruct(
                    batch_data, strategy="sample")

                for sample in decoded_batch:
                    string = ' '.join([str(s) for s in sample])
                    fout.write(f"{string}\n")

    def generate_samples(self):
        with open(fname, "w", encoding="utf-8") as fout:
            decoded_batch = model.decode(z, strategy="sample")

            for sample in decoded_batch:
                string = ' '.join([str(s) for s in sample])
                fout.write(f"{string}\n")
