import matplotlib.pyplot as plt
import torch

class SpectrumPlotter:
    def __init__(self, spec_calc, plot_config):
        self.calc = spec_calc
        self.plot_config = plot_config

    def display(self):
        freq = self.calc.freq
        mask = (freq >= self.plot_config.f_min) & (freq <= self.plot_config.f_max)
        freq_masked = freq[mask]

        for order in self.plot_config.display_orders:
            key = f"s{order}"
            spectrum_dict = getattr(self.calc, key, None)

            if spectrum_dict is None:
                print(f"Warning: No spectrum of order {key} found.")
                continue

            n_entries = len(spectrum_dict)
            n_panels = len(self.plot_config.plot_format)

            if order == 2:
                # --- S2: line plots stacked ---
                fig, axes = plt.subplots(
                    nrows=n_entries,
                    ncols=n_panels,
                    figsize=(6 * n_panels, 3.5 * n_entries),
                    squeeze=False,
                    sharex=True
                )

                for row_idx, (ops, data) in enumerate(spectrum_dict.items()):
                    label_base = f"({', '.join(ops)})"
                    data = data[mask]

                    for col_idx, fmt in enumerate(self.plot_config.plot_format):
                        ax = axes[row_idx][col_idx]

                        if fmt == "re":
                            y = data.real
                            ax.set_title(f"Re {label_base}")
                        elif fmt == "im":
                            y = data.imag
                            ax.set_title(f"Im {label_base}")
                        elif fmt == "abs":
                            y = torch.abs(torch.tensor(data))
                            ax.set_title(f"|{label_base}|")
                        else:
                            raise ValueError(f"Unknown plot_format: {fmt}")

                        ax.plot(freq_masked.cpu().numpy(), y)
                        ax.set_xlabel("ω")
                        ax.set_ylabel(f"S{order}")

                plt.tight_layout()
                plt.show()

            elif order == 3:
                # --- S3: stacked heatmaps in one figure ---
                freq_full = self.calc.freq
                extent = (freq_full[0], freq_full[-1], freq_full[0], freq_full[-1])

                fig, axes = plt.subplots(
                    nrows=n_entries,
                    ncols=n_panels,
                    figsize=(6 * n_panels, 3.5 * n_entries),
                    squeeze=False,
                    sharex=True,
                    sharey=True
                )

                for row_idx, (ops, data) in enumerate(spectrum_dict.items()):
                    label_base = f"({', '.join(ops)})"
                    data_real = torch.tensor(data.real)
                    data_imag = torch.tensor(data.imag)

                    for col_idx, fmt in enumerate(self.plot_config.plot_format):
                        ax = axes[row_idx][col_idx]

                        if fmt == "re":
                            y = data_real
                            vmax = torch.max(torch.abs(y)).item()
                            vmin = -vmax
                            title = f"Re {label_base}"
                        elif fmt == "im":
                            y = data_imag
                            vmax = torch.max(torch.abs(y)).item()
                            vmin = -vmax
                            title = f"Im {label_base}"
                        
                        else:
                            raise ValueError(f"Unknown plot_format: {fmt}")

                        im = ax.imshow(
                            y,
                            extent=extent,
                            origin='lower',
                            cmap='seismic',
                            aspect='auto',
                            vmin=vmin,
                            vmax=vmax
                        )
                        ax.set_title(title)
                        ax.set_xlabel("ω₁")
                        ax.set_ylabel("ω₂")
                        fig.colorbar(im, ax=ax)

                plt.tight_layout()
                plt.show()

            else:
                print(f"Order {order} is not yet supported.")
