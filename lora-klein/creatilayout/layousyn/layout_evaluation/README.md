Evaluations:

1. NSR: Spatial and Counting Evaluation
2. COCOGrounded FID


# Layout File Format
```json
{
        "iter": 0, // iteration number
        "object_list": [
            [
                "<object-name>",
                [
                    0.03125, // x coordinate of the top left corner
                    0.265625, // y coordinate of the top left corner
                    0.421875, // x coordinate of the bottom right corner
                    0.671875 // y coordinate of the bottom right corner
                ]
            ],
            [
                "<object-name>",
                [
                    0.5, // x coordinate of the top left corner
                    0.53125, // y coordinate of the top left corner
                    1.0, // x coordinate of the bottom right corner
                    0.875 // y coordinate of the bottom right corner
                ]
            ]
        ],
        "prompt": "<caption>",
        "query_id": "<query_id>",
    },
``` 