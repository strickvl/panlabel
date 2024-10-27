// This is where your library code goes
pub fn your_function() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        // Your tests
        assert_eq!(2 + 2, 4);
    }
}
