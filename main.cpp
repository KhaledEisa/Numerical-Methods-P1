#include <cstdio>
#include <chrono>

#define MAX 1000

// Multiply big integer (in res[]) by x
void multiply(int x, int res[], int *res_size) {
    int carry = 0;
    for (int i = 0; i < *res_size; i++) {
        int prod = res[i] * x + carry;
        res[i] = prod % 10;
        carry = prod / 10;
    }
    while (carry) {
        res[(*res_size)++] = carry % 10;
        carry /= 10;
    }
}

// Brute‐force factorial
void factorial_brute_force(int n) {
    int res[MAX];
    res[0] = 1;
    int res_size = 1;
    for (int i = 2; i <= n; i++)
        multiply(i, res, &res_size);

    std::printf("Brute Force Factorial of %d:\n", n);
    for (int i = res_size - 1; i >= 0; i--)
        std::printf("%d", res[i]);
    std::printf("\n");
}

// Transfer & Conquer factorial
void factorial_transfer_and_conquer(int n) {
    int res[MAX];
    res[0] = 1;
    int res_size = 1;

    int i;
    // Group 2×2 factors into one integer before invoking multiply()
    for (i = 2; i + 1 <= n; i += 2) {
        int pair = i * (i + 1);
        multiply(pair, res, &res_size);
    }
    // If n is odd, we still have one leftover factor at the end
    if (i == n) {
        multiply(i, res, &res_size);
    }

    // Print result
    printf("Transfer & Conquer Factorial of %d:\n", n);
    for (int j = res_size - 1; j >= 0; j--) {
        printf("%d", res[j]);
    }
    printf("\n");
}


// Helper for divide & conquer multiplication
void multiply_inner(int a[], int b[], int temp[], int i, int j, int b_size, int carry) {
    if (j >= b_size) {
        if (carry > 0) temp[i + b_size] += carry;
        return;
    }
    int sum = temp[i + j] + a[i] * b[j] + carry;
    temp[i + j] = sum % 10;
    carry = sum / 10;
    multiply_inner(a, b, temp, i, j + 1, b_size, carry);
}

void multiply_outer(int a[], int b[], int temp[], int i, int a_size, int b_size) {
    if (i >= a_size) return;
    multiply_inner(a, b, temp, i, 0, b_size, 0);
    multiply_outer(a, b, temp, i + 1, a_size, b_size);
}

void finalize_result(int temp[], int sz, int result[], int idx, int *result_size) {
    if (idx == sz) {
        *result_size = sz;
        return;
    }
    result[idx] = temp[idx];
    finalize_result(temp, sz, result, idx + 1, result_size);
}

void multiply_arrays(int a[], int a_size, int b[], int b_size, int result[], int *result_size) {
    int temp[MAX] = {0};
    multiply_outer(a, b, temp, 0, a_size, b_size);

    int sz = a_size + b_size;
    while (sz > 1 && temp[sz - 1] == 0) sz--;

    finalize_result(temp, sz, result, 0, result_size);
}

// Multiply single int x into result[]
void multiply_single(int x, int result[], int *result_size) {
    int carry = 0;
    for (int i = 0; i < *result_size; i++) {
        int prod = result[i] * x + carry;
        result[i] = prod % 10;
        carry = prod / 10;
    }
    while (carry) {
        result[(*result_size)++] = carry % 10;
        carry /= 10;
    }
}

// Factorial of single number n
void factorial_single(int n, int result[], int *result_size) {
    result[0] = 1;
    *result_size = 1;
    multiply_single(n, result, result_size);
}

// Divide & Conquer factorial on range [low..high]
void factorial_range(int low, int high, int result[], int *result_size) {
    if (low > high) {
        result[0] = 1;  *result_size = 1;
        return;
    }
    if (low == high) {
        factorial_single(low, result, result_size);
        return;
    }
    int mid = (low + high) / 2;
    int left[MAX], right[MAX];
    int left_size, right_size;

    factorial_range(low, mid, left, &left_size);
    factorial_range(mid + 1, high, right, &right_size);
    multiply_arrays(left, left_size, right, right_size, result, result_size);
}

// Print digits in reverse order
void print_result(int result[], int size) {
    for (int i = size - 1; i >= 0; i--)
        std::printf("%d", result[i]);
    std::printf("\n");
}

// Divide & Conquer factorial entry point
void factorial_divide_and_conquer(int n) {
    int result[MAX], result_size;
    factorial_range(1, n, result, &result_size);
    std::printf("Divide & Conquer Factorial of %d:\n", n);
    print_result(result, result_size);
}

int main() {
    using Clock = std::chrono::high_resolution_clock;
    using ms    = std::chrono::duration<double, std::milli>;

    int n;
    std::printf("Enter a number: ");
    if (std::scanf("%d", &n) != 1) return 0;

    // Time Brute‐Force
    {
        auto t0 = Clock::now();
        factorial_brute_force(n);
        auto t1 = Clock::now();
        std::printf("  → time: %.3f ms\n\n", ms(t1 - t0).count());
    }

    // Time Transfer & Conquer
    {
        auto t0 = Clock::now();
        factorial_transfer_and_conquer(n);
        auto t1 = Clock::now();
        std::printf("  → time: %.3f ms\n\n", ms(t1 - t0).count());
    }

    // Time Divide & Conquer
    {
        auto t0 = Clock::now();
        factorial_divide_and_conquer(n);
        auto t1 = Clock::now();
        std::printf("  → time: %.3f ms\n", ms(t1 - t0).count());
    }

    return 0;
}
