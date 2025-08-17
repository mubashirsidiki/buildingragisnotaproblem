"use client"
import { Moon, Sun } from "lucide-react"
import { useTheme } from "next-themes"
import { useEffect, useState } from "react"

export function ThemeToggle() {
  const { setTheme, theme, resolvedTheme } = useTheme()
  const [mounted, setMounted] = useState(false)

  // Avoid hydration mismatch
  useEffect(() => {
    setMounted(true)
    console.log("ThemeToggle mounted, theme:", theme, "resolvedTheme:", resolvedTheme)
  }, [theme, resolvedTheme])

  if (!mounted) {
    return (
      <button 
        className="w-9 h-8 px-0 rounded-md hover:bg-accent hover:text-accent-foreground transition-colors"
        suppressHydrationWarning
      >
        <Sun className="h-[1.2rem] w-[1.2rem] mx-auto" />
        <span className="sr-only">Toggle theme</span>
      </button>
    )
  }

  const currentTheme = resolvedTheme || theme

  const toggleTheme = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    console.log("Theme toggle clicked! Current theme:", currentTheme)
    console.log("Event target:", e.target)
    console.log("Event currentTarget:", e.currentTarget)
    
    if (currentTheme === "dark") {
      console.log("Setting theme to light")
      setTheme("light")
    } else {
      console.log("Setting theme to dark")
      setTheme("dark")
    }
  }

  return (
    <button 
      className="w-9 h-8 px-0 rounded-md hover:bg-accent hover:text-accent-foreground transition-colors cursor-pointer border-none bg-transparent text-foreground"
      onClick={toggleTheme}
      aria-label="Toggle theme"
      type="button"
      suppressHydrationWarning
      style={{ 
        pointerEvents: 'auto',
        position: 'relative',
        zIndex: 10,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}
    >
      {currentTheme === "dark" ? (
        <Sun className="h-[1.2rem] w-[1.2rem] transition-all duration-200" />
      ) : (
        <Moon className="h-[1.2rem] w-[1.2rem] transition-all duration-200" />
      )}
      <span className="sr-only">Toggle theme</span>
    </button>
  )
}
