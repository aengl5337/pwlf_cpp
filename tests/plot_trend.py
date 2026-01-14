import matplotlib.pyplot as plt

def read_data(filename):
    with open(filename, 'r') as f:
        # Read lines, strip whitespace, skip empty lines
        return [float(line.strip()) for line in f if line.strip()]

def main():
    # Load the raw S&P 500 data
    raw_data = read_data('snp500.txt')

    # Load your filtered result
    trend_data = read_data('result.txt')

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(raw_data, color='lightgray', label='Raw S&P 500 (Noisy)', linewidth=1)
    plt.plot(trend_data, color='blue', label='L1 Trend Filter (Lambda=50)', linewidth=2)

    plt.title('L1 Trend Filtering: S&P 500')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == '__main__':
    main()