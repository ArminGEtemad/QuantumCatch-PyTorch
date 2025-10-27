import matplotlib.pyplot as plt
import torch

class SpectrumPlotter:
    def __init__(self, spec_calc, plot_config):
        self.calc = spec_calc
        self.plot_config = plot_config

    def display(self):
        freq = self.calc.freq
        mask = (freq >= self.plot_config.f_min) & (freq <= self.plot_config.f_max)
        freq = freq[mask]

        for order in self.plot_config.display_orders:
            key = f"s{order}"
            spectrum_dict = getattr(self.calc, key, None)

            if spectrum_dict is None:
                print(f"Warning: No spectrum of order {key} found.")
                continue

            n_pairs = len(spectrum_dict)
            n_panels = len(self.plot_config.plot_format)

            fig, axes = plt.subplots(
                nrows=n_pairs,
                ncols=n_panels,
                figsize=(6 * n_panels, 3.5 * n_pairs),
                sharex=True
            )

            if n_pairs == 1 and n_panels == 1:
                axes = [[axes]]
            elif n_pairs == 1:
                axes = [axes]
            elif n_panels == 1:
                axes = [[ax] for ax in axes]
            for row_idx, ((op0, op1), data) in enumerate(spectrum_dict.items()):
                data = data[mask]
                label_base = f"({op0}, {op1})"

                for col_idx, fmt in enumerate(self.plot_config.plot_format):
                    ax = axes[row_idx][col_idx]

                    if fmt == "re":
                        y = data.real
                        ax.set_title(f"Re {label_base}")
                    elif fmt == "im":
                        y = data.imag
                        ax.set_title(f"Im {label_base}")
                    elif fmt == "abs":
                        y = torch.abs(data)
                        ax.set_title(f"|{label_base}|")
                    else:
                        raise ValueError(f"Unknown plot_format: {fmt}")

                    ax.plot(freq.cpu().numpy(), y)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
