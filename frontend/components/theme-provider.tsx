"use client"
import { ThemeProvider as NextThemesProvider } from "next-themes"
import type { ThemeProviderProps } from "next-themes"
import { useEffect, useState } from "react"

export function ThemeProvider({ children, ...props }: ThemeProviderProps) {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
    console.log("ThemeProvider mounted with props:", props)
  }, [props])

  if (!mounted) {
    return <div suppressHydrationWarning>{children}</div>
  }

  return (
    <NextThemesProvider 
      {...props}
      onThemeChange={(theme) => {
        console.log("Theme changed to:", theme)
      }}
    >
      {children}
    </NextThemesProvider>
  )
}
