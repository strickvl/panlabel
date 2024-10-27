use assert_cmd::Command;

#[test]
fn runs() {
    let mut cmd = Command::cargo_bin("panlabel").unwrap();
    cmd.assert().success();
}

#[test]
fn outputs_tool_name() {
    let mut cmd = Command::cargo_bin("panlabel").unwrap();
    cmd.arg("-V");
    cmd.assert().success().stdout("panlabel 0.1.0\n");
}
