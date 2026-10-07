# Prompts

This directory contains the prompt template used in the experiments reported in the paper.

## Repair Prompt

The following prompt format is used for decoder-only models:

```
<bug_method>
{buggy_method}
<fix_code>
```

where `{buggy_method}` denotes the buggy method provided as input. The model is expected to generate the repaired code after `<fix_code>`.

## Example

Input:

```
<bug_method>
public LegendItemCollection getLegendItems() {
    CategoryDataset dataset = this.plot.getDataset(index);

// bug_start
    if (dataset != null) {
// bug_end
        return result;
    }

    int seriesCount = dataset.getRowCount();
}
<fix_code>
```

Expected output:

```
// fix_start
if (dataset == null) {
// fix_end
```

During evaluation, only the generated content following `<fix_code>` is treated as the predicted repair.