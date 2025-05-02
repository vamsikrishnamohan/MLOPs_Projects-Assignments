# Dataset versions to test
$versions = @("v1", "v2", "v3", "v1+v2", "v1+v2+v3")

# Random seeds to test
$seeds = @(42, 123, 456)

# Run all combinations
foreach ($version in $versions) {
    foreach ($seed in $seeds) {
        Write-Host "Running experiment: version=$version, seed=$seed"
        dvc exp run -S data.version="$version" -S seed="$seed" -n "${version}-seed-${seed}"
    }
}

# Show results
dvc exp show --no-pager