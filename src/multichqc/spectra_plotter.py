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

            is_s3 = order == 3
            n_entries = len(spectrum_dict)
            n_panels = len(self.plot_config.plot_format)

            fig, axes = plt.subplots(
                nrows=n_entries,
                ncols=n_panels,
                figsize=(6 * n_panels, 3.5 * n_entries),
                squeeze=False
            )

            for row_idx, (ops, data) in enumerate(spectrum_dict.items()):
                if is_s3:

                    data = data[mask][:, mask]
                    label_base = f"({ops[0]}, {ops[1]}, {ops[2]})"
                else:
                    data = data[mask]
                    label_base = f"({ops[0]}, {ops[1]})"

                for col_idx, fmt in enumerate(self.plot_config.plot_format):
                    ax = axes[row_idx][col_idx]

                    if fmt == "re":
                        y = data.real
                        title = f"Re {label_base}"
                    elif fmt == "im":
                        y = data.imag
                        title = f"Im {label_base}"
                    elif fmt == "abs":
                        y = torch.abs(torch.tensor(data)) if not isinstance(data, torch.Tensor) else torch.abs(data)
                        title = f"|{label_base}|"
                    else:
                        raise ValueError(f"Unknown plot_format: {fmt}")

                    ax.set_title(title)

                    if is_s3:
                        vmax = torch.max(torch.abs(torch.tensor(y))).item()
                        vmin = -vmax

                        im = ax.imshow(
                            y,
                            extent=(freq[0], freq[-1], freq[0], freq[-1]),
                            origin='lower',
                            aspect='auto',
                            cmap='seismic',
                            vmin=vmin,
                            vmax=vmax
                        )
                        ax.set_xlabel("f")
                        ax.set_ylabel("f")
                        fig.colorbar(im, ax=ax)
                    else:
                        ax.plot(freq.cpu().numpy(), y)
                        ax.set_xlabel("f")
                        ax.set_ylabel("s2")

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()