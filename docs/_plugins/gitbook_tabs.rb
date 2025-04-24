module Jekyll
  class GitBookTabs < Liquid::Block
    def render(context)
      # Simply return the content without processing
      super
    end
  end

  class GitBookTab < Liquid::Block
    def initialize(tag_name, markup, tokens)
      super
      @markup = markup
    end

    def render(context)
      # Simply return the content without processing
      super
    end
  end
end

Liquid::Template.register_tag('tabs', Jekyll::GitBookTabs)
Liquid::Template.register_tag('tab', Jekyll::GitBookTab) 
