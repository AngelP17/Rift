"use client";

import { animate, motion, useMotionValue, useTransform } from "framer-motion";
import { useEffect } from "react";

type AnimatedNumberProps = {
  value: number;
  format: (value: number) => string;
  className?: string;
};

export function AnimatedNumber({ value, format, className }: AnimatedNumberProps) {
  const motionValue = useMotionValue(value);
  const rounded = useTransform(motionValue, (latest) => format(latest));

  useEffect(() => {
    const controls = animate(motionValue, value, {
      duration: 1.1,
      ease: [0.25, 0.46, 0.45, 0.94]
    });

    return () => controls.stop();
  }, [motionValue, value]);

  return <motion.span className={className}>{rounded}</motion.span>;
}
