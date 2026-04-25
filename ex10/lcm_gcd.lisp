(defun my-gcd (a b)
  (if (= b 0)
      a
      (my-gcd b (mod a b))
  )
)

(defun my-lcm (a b)
  (/ (* a b) (my-gcd a b))
)

(let ((a 0) (b 0))
  (write-line "Enter first number:")
  (setq a (read))

  (write-line "Enter second number:")
  (setq b (read))

  (print (list "GCD =" (my-gcd a b)))
  (print (list "LCM =" (my-lcm a b)))
)
