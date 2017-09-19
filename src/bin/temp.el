(defun njm/setup-extended-500-file (number)
  (interactive)

  ;; open relevant file
  (find-file (format "run_infShieldState_extended_500_%02d_18.cpp" number))

  ;; get point into position
  (goto-char (point-min))
  (re-search-forward "setup networks")

  ;; comment out 100s
  (re-search-forward "{")
  (forward-line 0)
  (set-mark-command nil)
  (re-search-forward "}")
  (comment-dwim-2)
  (deactivate-mark)

  ;; comment out 1000s
  (re-search-forward "{")
  (re-search-forward "{")
  (forward-line 0)
  (set-mark-command nil)
  (re-search-forward "}")
  (comment-dwim-2)
  (deactivate-mark)

  ;; save as new file
  (write-file (format "run_infShieldState_extended_500_%02d_18.cpp" number))
  (kill-buffer)
  )

(defun njm/setup-extended-1000-file (number)
  (interactive)

  ;; open relevant file
  (find-file (format "run_infShieldState_extended_1000_%02d_18.cpp" number))

  ;; get point into position
  (goto-char (point-min))
  (re-search-forward "setup networks")

  ;; comment out 100s
  (re-search-forward "{")
  (forward-line 0)
  (set-mark-command nil)
  (re-search-forward "}")
  (comment-dwim-2)
  (deactivate-mark)

  ;; comment out 500s
  (re-search-forward "{")
  (forward-line 0)
  (set-mark-command nil)
  (re-search-forward "}")
  (comment-dwim-2)
  (deactivate-mark)

  ;; save as new file
  (write-file (format "run_infShieldState_extended_1000_%02d_18.cpp" number))
  (kill-buffer)
  )

(defun njm/setup-extended ()
  (interactive)
  (njm/setup-extended-500-file 2)
  (dotimes (number 18)
    (njm/setup-extended-500-file (+ number 1))
    (njm/setup-extended-1000-file (+ number 1))
    )
  )
