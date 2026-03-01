use panlabel::ir::io_json::{from_json_str, to_json_string};
use proptest::prelude::*;

mod proptest_helpers;

proptest! {
    #![proptest_config(proptest_helpers::proptest_config())]

    #[test]
    fn ir_json_roundtrip_is_lossless(dataset in proptest_helpers::arb_dataset_full(5, 5, 20)) {
        let json = to_json_string(&dataset).expect("serialize ir json");
        let restored = from_json_str(&json).expect("parse ir json");

        prop_assert_eq!(dataset, restored);
    }

    #[test]
    fn ir_json_roundtrip_is_idempotent(dataset in proptest_helpers::arb_dataset_full(5, 5, 20)) {
        let first_json = to_json_string(&dataset).expect("serialize first pass");
        let first = from_json_str(&first_json).expect("parse first pass");

        let second_json = to_json_string(&first).expect("serialize second pass");
        let second = from_json_str(&second_json).expect("parse second pass");

        prop_assert_eq!(first, second);
    }
}
